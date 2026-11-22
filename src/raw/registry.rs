use std::mem;

use crate::raw::build::BuilderNode;
use crate::raw::{CompiledAddr, Transition, NONE_ADDRESS};

#[derive(Debug)]
pub struct Registry {
    table: Vec<RegistryCell>,
    table_size: usize, // number of rows
    mru_size: usize,   // number of columns
    /// The number of cache misses, i.e., the number of nodes that had to be
    /// compiled and written out. This is used as a proxy for the size of
    /// the working set.
    inserts: usize,
    /// The maximum number of cells this registry may grow to hold. This
    /// bounds the memory used by the registry.
    max_cells: usize,
}

#[derive(Debug)]
struct RegistryCache<'a> {
    cells: &'a mut [RegistryCell],
}

#[derive(Clone, Debug)]
pub struct RegistryCell {
    addr: CompiledAddr,
    node: BuilderNode,
}

#[derive(Debug)]
pub enum RegistryEntry<'a> {
    Found(CompiledAddr),
    NotFound(&'a mut RegistryCell),
    Rejected,
}

impl Registry {
    /// Create a registry with a fixed capacity of `table_size * mru_size`
    /// cells. A fixed capacity registry never grows.
    pub fn new(table_size: usize, mru_size: usize) -> Registry {
        let max_cells = table_size.checked_mul(mru_size).unwrap();
        Registry::with_capacity(table_size, mru_size, max_cells)
    }

    /// Create a registry that grows with the size of the working set, but
    /// whose memory usage is bounded by (approximately) `max_bytes`.
    ///
    /// The registry starts out small (`initial_table_size * mru_size`
    /// cells) and doubles in size whenever the number of compiled nodes
    /// exceeds the number of cells, until the memory budget is reached.
    pub fn with_max_bytes(
        initial_table_size: usize,
        mru_size: usize,
        max_bytes: usize,
    ) -> Registry {
        // The approximate cost of a single occupied cell: the cell itself,
        // plus the transitions stored out of line in the node's `Vec`
        // (including over-allocation and allocator overhead). Measured at
        // ~140 bytes per occupied cell on real world data.
        let per_cell =
            mem::size_of::<RegistryCell>() + 4 * mem::size_of::<Transition>();
        let max_cells = max_bytes / per_cell;
        Registry::with_capacity(initial_table_size, mru_size, max_cells)
    }

    fn with_capacity(
        table_size: usize,
        mru_size: usize,
        max_cells: usize,
    ) -> Registry {
        let empty_cell = RegistryCell::none();
        let ncells = table_size.checked_mul(mru_size).unwrap();
        Registry {
            table: vec![empty_cell; ncells],
            table_size,
            mru_size,
            inserts: 0,
            max_cells: max_cells.max(ncells),
        }
    }

    pub fn entry<'a>(&'a mut self, node: &BuilderNode) -> RegistryEntry<'a> {
        if self.table.is_empty() {
            return RegistryEntry::Rejected;
        }
        self.maybe_grow();
        let bucket = self.hash(node);
        let start = self.mru_size * bucket;
        let end = start + self.mru_size;
        let entry =
            RegistryCache { cells: &mut self.table[start..end] }.entry(node);
        if matches!(&entry, RegistryEntry::NotFound(_)) {
            self.inserts += 1;
        }
        entry
    }

    fn hash(&self, node: &BuilderNode) -> usize {
        hash_node(node, self.table_size)
    }

    /// Double the size of the table if the working set (the number of
    /// compiled nodes) has outgrown it, without exceeding the memory
    /// budget.
    fn maybe_grow(&mut self) {
        if self.inserts < self.table.len()
            || self.table.len() >= self.max_cells
        {
            return;
        }
        let new_table_size =
            (self.table_size * 2).min(self.max_cells / self.mru_size);
        if new_table_size <= self.table_size {
            return;
        }
        let mut new_table =
            vec![RegistryCell::none(); new_table_size * self.mru_size];
        for cell in self.table.drain(..) {
            if cell.is_none() {
                continue;
            }
            let bucket = hash_node(&cell.node, new_table_size);
            let start = self.mru_size * bucket;
            // Place the cell in the first empty slot of its bucket, which
            // preserves MRU order. If the bucket is full, then drop the
            // cell. Losing a cached node is always safe: it just means the
            // node will be written out again if it's seen again.
            for slot in &mut new_table[start..start + self.mru_size] {
                if slot.is_none() {
                    *slot = cell;
                    break;
                }
            }
        }
        self.table = new_table;
        self.table_size = new_table_size;
    }
}

/// Hash a node to a bucket in a table with `table_size` rows.
fn hash_node(node: &BuilderNode, table_size: usize) -> usize {
    // Basic FNV-1a hash as described:
    // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
    //
    // In unscientific experiments, this provides the same compression
    // as `std::hash::SipHasher` but is much much faster.
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = 14695981039346656037;
    h = (h ^ (node.is_final as u64)).wrapping_mul(FNV_PRIME);
    h = (h ^ node.final_output.value()).wrapping_mul(FNV_PRIME);
    for t in &node.trans {
        h = (h ^ (t.inp as u64)).wrapping_mul(FNV_PRIME);
        h = (h ^ t.out.value()).wrapping_mul(FNV_PRIME);
        h = (h ^ (t.addr as u64)).wrapping_mul(FNV_PRIME);
    }
    (h as usize) % table_size
}

impl<'a> RegistryCache<'a> {
    fn entry(mut self, node: &BuilderNode) -> RegistryEntry<'a> {
        if self.cells.len() == 1 {
            let cell = &mut self.cells[0];
            if !cell.is_none() && &cell.node == node {
                RegistryEntry::Found(cell.addr)
            } else {
                cell.node.clone_from(node);
                RegistryEntry::NotFound(cell)
            }
        } else if self.cells.len() == 2 {
            let cell1 = &mut self.cells[0];
            if !cell1.is_none() && &cell1.node == node {
                return RegistryEntry::Found(cell1.addr);
            }

            let cell2 = &mut self.cells[1];
            if !cell2.is_none() && &cell2.node == node {
                let addr = cell2.addr;
                self.cells.swap(0, 1);
                return RegistryEntry::Found(addr);
            }

            self.cells[1].node.clone_from(node);
            self.cells.swap(0, 1);
            RegistryEntry::NotFound(&mut self.cells[0])
        } else {
            let find = |c: &RegistryCell| !c.is_none() && &c.node == node;
            if let Some(i) = self.cells.iter().position(find) {
                let addr = self.cells[i].addr;
                self.promote(i); // most recently used
                RegistryEntry::Found(addr)
            } else {
                let last = self.cells.len() - 1;
                self.cells[last].node.clone_from(node); // discard LRU
                self.promote(last);
                RegistryEntry::NotFound(&mut self.cells[0])
            }
        }
    }

    fn promote(&mut self, mut i: usize) {
        assert!(i < self.cells.len());
        while i > 0 {
            self.cells.swap(i - 1, i);
            i -= 1;
        }
    }
}

impl RegistryCell {
    fn none() -> RegistryCell {
        RegistryCell { addr: NONE_ADDRESS, node: BuilderNode::default() }
    }

    fn is_none(&self) -> bool {
        self.addr == NONE_ADDRESS
    }

    pub fn insert(&mut self, addr: CompiledAddr) {
        self.addr = addr;
    }
}

#[cfg(test)]
mod tests {
    use super::{Registry, RegistryCache, RegistryCell, RegistryEntry};
    use crate::raw::build::BuilderNode;
    use crate::raw::{Output, Transition};

    fn assert_rejected(entry: RegistryEntry) {
        match entry {
            RegistryEntry::Rejected => {}
            entry => panic!("expected rejected entry, got: {:?}", entry),
        }
    }

    fn assert_not_found(entry: RegistryEntry) {
        match entry {
            RegistryEntry::NotFound(_) => {}
            entry => panic!("expected nout found entry, got: {:?}", entry),
        }
    }

    fn assert_insert_and_found(reg: &mut Registry, bnode: &BuilderNode) {
        match reg.entry(&bnode) {
            RegistryEntry::NotFound(cell) => cell.insert(1234),
            entry => panic!("unexpected not found entry, got: {:?}", entry),
        }
        match reg.entry(&bnode) {
            RegistryEntry::Found(addr) => assert_eq!(addr, 1234),
            entry => panic!("unexpected found entry, got: {:?}", entry),
        }
    }

    #[test]
    fn empty_is_ok() {
        let mut reg = Registry::new(0, 0);
        let bnode = BuilderNode {
            is_final: false,
            final_output: Output::zero(),
            trans: vec![],
        };
        assert_rejected(reg.entry(&bnode));
    }

    #[test]
    fn one_final_is_ok() {
        let mut reg = Registry::new(1, 1);
        let bnode = BuilderNode {
            is_final: true,
            final_output: Output::zero(),
            trans: vec![],
        };
        assert_insert_and_found(&mut reg, &bnode);
    }

    #[test]
    fn one_with_trans_is_ok() {
        let mut reg = Registry::new(1, 1);
        let bnode = BuilderNode {
            is_final: false,
            final_output: Output::zero(),
            trans: vec![Transition {
                addr: 0,
                inp: b'a',
                out: Output::zero(),
            }],
        };
        assert_insert_and_found(&mut reg, &bnode);
        assert_not_found(
            reg.entry(&BuilderNode { is_final: true, ..bnode.clone() }),
        );
        assert_not_found(reg.entry(&BuilderNode {
            trans: vec![Transition {
                addr: 0,
                inp: b'b',
                out: Output::zero(),
            }],
            ..bnode.clone()
        }));
        assert_not_found(reg.entry(&BuilderNode {
            trans: vec![Transition {
                addr: 0,
                inp: b'a',
                out: Output::new(1),
            }],
            ..bnode.clone()
        }));
    }

    #[test]
    fn cache_works() {
        let mut reg = Registry::new(1, 1);

        let bnode1 = BuilderNode { is_final: true, ..BuilderNode::default() };
        assert_insert_and_found(&mut reg, &bnode1);

        let bnode2 =
            BuilderNode { final_output: Output::new(1), ..bnode1.clone() };
        assert_insert_and_found(&mut reg, &bnode2);
        assert_not_found(reg.entry(&bnode1));
    }

    #[test]
    fn promote() {
        let bn = BuilderNode::default();
        let mut bnodes = vec![
            RegistryCell { addr: 1, node: bn.clone() },
            RegistryCell { addr: 2, node: bn.clone() },
            RegistryCell { addr: 3, node: bn.clone() },
            RegistryCell { addr: 4, node: bn.clone() },
        ];
        let mut cache = RegistryCache { cells: &mut bnodes };

        cache.promote(0);
        assert_eq!(cache.cells[0].addr, 1);
        assert_eq!(cache.cells[1].addr, 2);
        assert_eq!(cache.cells[2].addr, 3);
        assert_eq!(cache.cells[3].addr, 4);

        cache.promote(1);
        assert_eq!(cache.cells[0].addr, 2);
        assert_eq!(cache.cells[1].addr, 1);
        assert_eq!(cache.cells[2].addr, 3);
        assert_eq!(cache.cells[3].addr, 4);

        cache.promote(3);
        assert_eq!(cache.cells[0].addr, 4);
        assert_eq!(cache.cells[1].addr, 2);
        assert_eq!(cache.cells[2].addr, 1);
        assert_eq!(cache.cells[3].addr, 3);

        cache.promote(2);
        assert_eq!(cache.cells[0].addr, 1);
        assert_eq!(cache.cells[1].addr, 4);
        assert_eq!(cache.cells[2].addr, 2);
        assert_eq!(cache.cells[3].addr, 3);
    }

    #[test]
    fn grows_with_working_set() {
        let mut reg = Registry::with_max_bytes(2, 1, 1 << 20);
        let mut bnodes = vec![];
        for i in 0..1000usize {
            let bnode = BuilderNode {
                is_final: true,
                final_output: Output::new(i as u64),
                trans: vec![],
            };
            match reg.entry(&bnode) {
                RegistryEntry::NotFound(cell) => cell.insert(i + 2),
                entry => panic!("unexpected entry, got: {:?}", entry),
            }
            bnodes.push(bnode);
        }
        // The registry should have grown to hold every node inserted.
        for (i, bnode) in bnodes.iter().enumerate() {
            match reg.entry(bnode) {
                RegistryEntry::Found(addr) => assert_eq!(addr, i + 2),
                entry => panic!("unexpected entry, got: {:?}", entry),
            }
        }
    }

    #[test]
    fn growth_respects_budget() {
        use std::mem;

        use crate::raw::Transition;

        let per_cell =
            mem::size_of::<RegistryCell>() + 4 * mem::size_of::<Transition>();
        let mut reg = Registry::with_max_bytes(2, 1, 16 * per_cell);
        for i in 0..1000usize {
            let bnode = BuilderNode {
                is_final: true,
                final_output: Output::new(i as u64),
                trans: vec![],
            };
            match reg.entry(&bnode) {
                RegistryEntry::NotFound(cell) => cell.insert(i + 2),
                entry => panic!("unexpected entry, got: {:?}", entry),
            }
        }
        assert!(reg.table.len() <= 16);
    }
}
