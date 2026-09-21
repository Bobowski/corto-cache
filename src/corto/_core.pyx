# cython: language_level=3
"""Window-TinyLFU cache: admission window + SLRU main + 4-bit Count-Min sketch."""

from collections import namedtuple
from cpython.object cimport PyObject, PyObject_RichCompareBool, Py_EQ
from cpython.tuple cimport PyTuple_CheckExact, PyTuple_GET_ITEM, PyTuple_GET_SIZE
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.stdlib cimport calloc, free, malloc
from libc.string cimport memset


cdef extern from "Python.h":
    ctypedef struct PyMutex:
        uint8_t _bits
    void PyMutex_Lock(PyMutex *m) noexcept nogil
    void PyMutex_Unlock(PyMutex *m) noexcept nogil


cdef enum Region:
    WINDOW = 0
    PROBATION = 1
    PROTECTED = 2


cdef inline uint64_t _mix(uint64_t z) noexcept nogil:
    z = (z ^ (z >> 30)) * <uint64_t>0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) * <uint64_t>0x94d049bb133111eb
    return z ^ (z >> 31)


cdef inline Py_ssize_t _next_pow2(Py_ssize_t n) noexcept:
    cdef Py_ssize_t p = 1
    if n < 1:
        return 1
    while p < n:
        p <<= 1
    return p


cdef inline uint64_t _hash2(object a, object b):
    cdef uint64_t ha = <uint64_t><Py_hash_t>hash(a)
    cdef uint64_t hb = <uint64_t><Py_hash_t>hash(b)
    return _mix(ha ^ (hb + <uint64_t>0x9E3779B97F4A7C15))


cdef inline uint64_t _hash3(object a, object b, object c):
    cdef uint64_t ha = <uint64_t><Py_hash_t>hash(a)
    cdef uint64_t hb = <uint64_t><Py_hash_t>hash(b)
    cdef uint64_t hc = <uint64_t><Py_hash_t>hash(c)
    return _mix(ha ^ (hb + <uint64_t>0x9E3779B97F4A7C15)) ^ (
        hc * <uint64_t>0xBF58476D1CE4E5B9
    )


cdef inline bint _same(object left, object right) except -1:
    cdef int cmp
    if <PyObject*>left == <PyObject*>right:
        return 1
    cmp = PyObject_RichCompareBool(left, right, Py_EQ)
    if cmp == -1:
        raise
    return cmp == 1


cdef inline bint _tuple2_match(object key, object a, object b) except -1:
    if not PyTuple_CheckExact(key) or PyTuple_GET_SIZE(key) != 2:
        return 0
    return _same(<object>PyTuple_GET_ITEM(key, 0), a) and _same(
        <object>PyTuple_GET_ITEM(key, 1), b
    )


cdef inline bint _tuple3_match(object key, object a, object b, object c) except -1:
    if not PyTuple_CheckExact(key) or PyTuple_GET_SIZE(key) != 3:
        return 0
    return (
        _same(<object>PyTuple_GET_ITEM(key, 0), a)
        and _same(<object>PyTuple_GET_ITEM(key, 1), b)
        and _same(<object>PyTuple_GET_ITEM(key, 2), c)
    )


cdef struct Node:
    PyObject* key
    PyObject* value
    Node* prev
    Node* next
    uint8_t region
    uint64_t khash
    Py_ssize_t slot


cdef struct Slot:
    uint64_t khash
    Node* node


cdef Node* _TOMB = <Node*>1


cdef inline Node* _sentinel_new() except NULL:
    cdef Node* node = <Node*>calloc(1, sizeof(Node))
    if node == NULL:
        raise MemoryError()
    node.prev = node
    node.next = node
    return node


cdef inline void _link_mru(Node* sentinel, Node* node) noexcept nogil:
    cdef Node* last = sentinel.prev
    node.next = sentinel
    node.prev = last
    last.next = node
    sentinel.prev = node


cdef inline void _unlink(Node* node) noexcept nogil:
    node.prev.next = node.next
    node.next.prev = node.prev


cdef inline void _push(Node* sentinel, Node* node, uint8_t region, Py_ssize_t* size) noexcept nogil:
    node.region = region
    _link_mru(sentinel, node)
    size[0] += 1


cdef inline void _remove(Node* node, Py_ssize_t* size) noexcept nogil:
    _unlink(node)
    size[0] -= 1


cdef inline Node* _pop_lru(Node* sentinel, Py_ssize_t* size) noexcept nogil:
    cdef Node* node = sentinel.next
    if node == sentinel:
        return NULL
    _unlink(node)
    size[0] -= 1
    return node


cdef inline Node* _peek_lru(Node* sentinel) noexcept nogil:
    cdef Node* node = sentinel.next
    if node == sentinel:
        return NULL
    return node


cdef class FrequencySketch:
    """4-bit Count-Min sketch. About 8 bytes of counters per cache slot."""

    cdef uint64_t* table
    cdef Py_ssize_t nwords
    cdef uint32_t sample_size
    cdef uint32_t samples
    cdef uint32_t counter_mask

    def __cinit__(self, Py_ssize_t capacity):
        cdef Py_ssize_t ncounters
        if capacity < 8:
            capacity = 8
        ncounters = _next_pow2(capacity * 4)
        self.nwords = ncounters // 16
        self.table = <uint64_t*>calloc(<size_t>self.nwords, sizeof(uint64_t))
        if self.table == NULL:
            raise MemoryError()
        self.counter_mask = <uint32_t>(ncounters - 1)
        if capacity * 10 > 0x7FFFFFFF:
            self.sample_size = <uint32_t>0x7FFFFFFF
        else:
            self.sample_size = <uint32_t>(capacity * 10)
        self.samples = 0

    def __dealloc__(self):
        if self.table != NULL:
            free(self.table)
            self.table = NULL

    cdef inline int _get(self, uint32_t idx) noexcept nogil:
        cdef uint64_t word = self.table[idx >> 4]
        cdef uint32_t shift = (idx & 15) << 2
        return <int>((word >> shift) & 0xF)

    cdef inline void _bump(self, uint32_t idx) noexcept nogil:
        cdef uint32_t shift = (idx & 15) << 2
        cdef uint32_t wi = idx >> 4
        cdef uint64_t word = self.table[wi]
        if ((word >> shift) & 0xF) < 15:
            self.table[wi] = word + (<uint64_t>1 << shift)

    cdef inline void increment_hash(self, uint64_t h) noexcept nogil:
        cdef uint64_t a, b
        self.samples += 1
        if self.samples >= self.sample_size:
            self._age()
        a = _mix(h)
        b = _mix(a)
        self._bump(<uint32_t>a & self.counter_mask)
        self._bump(<uint32_t>(a >> 32) & self.counter_mask)
        self._bump(<uint32_t>b & self.counter_mask)
        self._bump(<uint32_t>(b >> 32) & self.counter_mask)

    cdef inline int frequency_hash(self, uint64_t h) noexcept nogil:
        cdef uint64_t a = _mix(h)
        cdef uint64_t b = _mix(a)
        cdef int est, cur
        est = self._get(<uint32_t>a & self.counter_mask)
        cur = self._get(<uint32_t>(a >> 32) & self.counter_mask)
        if cur < est:
            est = cur
        cur = self._get(<uint32_t>b & self.counter_mask)
        if cur < est:
            est = cur
        cur = self._get(<uint32_t>(b >> 32) & self.counter_mask)
        if cur < est:
            est = cur
        return est

    cdef int frequency(self, object key):
        return self.frequency_hash(<uint64_t><Py_hash_t>hash(key))

    cdef void _age(self) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(self.nwords):
            self.table[i] = (self.table[i] >> 1) & <uint64_t>0x7777777777777777
        self.samples = 0


cdef object _MISSING = object()


cdef class Cache:
    """Bounded mapping with Window-TinyLFU admission.

    New keys enter a small LRU window (1% of ``maxsize``). A window victim
    enters the SLRU main space only when the Count-Min sketch says it is
    more frequent than the main-space victim. One ``PyMutex`` guards every
    public method. A key or value ``__del__`` must leave this cache unchanged.
    """

    cdef Slot* _slots
    cdef Py_ssize_t _nslots
    cdef Py_ssize_t _slot_mask
    cdef Py_ssize_t _size
    cdef Py_ssize_t _tombs
    cdef Node* _window_s
    cdef Node* _probation_s
    cdef Node* _protected_s
    cdef Py_ssize_t _window_n
    cdef Py_ssize_t _probation_n
    cdef Py_ssize_t _protected_n
    cdef FrequencySketch _sketch
    cdef Py_ssize_t _maxsize
    cdef Py_ssize_t _window_max
    cdef Py_ssize_t _main_max
    cdef Py_ssize_t _protected_max
    cdef Py_ssize_t _hits
    cdef Py_ssize_t _misses
    cdef dict _keep
    cdef Node* _free_head
    cdef PyMutex _mu

    def __cinit__(self, Py_ssize_t maxsize):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if maxsize > (1 << 30):
            raise ValueError("maxsize is too large")
        self._maxsize = maxsize
        self._keep = {}
        self._window_max = maxsize // 100
        if self._window_max < 1:
            self._window_max = 1
        self._main_max = maxsize - self._window_max
        self._protected_max = <Py_ssize_t>(self._main_max * 4 // 5)
        self._nslots = _next_pow2(maxsize * 2)
        if self._nslots < 8:
            self._nslots = 8
        self._slot_mask = self._nslots - 1
        self._slots = <Slot*>calloc(<size_t>self._nslots, sizeof(Slot))
        if self._slots == NULL:
            raise MemoryError()
        self._size = 0
        self._tombs = 0
        self._window_s = _sentinel_new()
        self._probation_s = _sentinel_new()
        self._protected_s = _sentinel_new()
        self._window_n = 0
        self._probation_n = 0
        self._protected_n = 0
        self._sketch = FrequencySketch(maxsize)
        self._hits = 0
        self._misses = 0
        self._free_head = NULL
        self._mu._bits = 0

    cdef inline void _lock(self) noexcept nogil:
        PyMutex_Lock(&self._mu)

    cdef inline void _unlock(self) noexcept nogil:
        PyMutex_Unlock(&self._mu)

    cdef inline Node* _node_acquire(
        self, object key, object value, uint64_t khash
    ) except NULL:
        cdef Node* node = self._free_head
        if node != NULL:
            self._free_head = node.next
        else:
            node = <Node*>malloc(sizeof(Node))
            if node == NULL:
                raise MemoryError()
        node.key = <PyObject*>key
        node.value = <PyObject*>value
        node.prev = NULL
        node.next = NULL
        node.region = WINDOW
        node.khash = khash
        node.slot = -1
        return node

    cdef inline void _node_release(self, Node* node) noexcept:
        node.key = NULL
        node.value = NULL
        node.prev = NULL
        node.slot = -1
        node.next = self._free_head
        self._free_head = node

    cdef void _free_freelist(self) noexcept:
        cdef Node* node = self._free_head
        cdef Node* nxt
        while node != NULL:
            nxt = node.next
            free(node)
            node = nxt
        self._free_head = NULL

    def __dealloc__(self):
        self._discard_nodes()
        self._free_freelist()
        if self._window_s != NULL:
            free(self._window_s)
            self._window_s = NULL
        if self._probation_s != NULL:
            free(self._probation_s)
            self._probation_s = NULL
        if self._protected_s != NULL:
            free(self._protected_s)
            self._protected_s = NULL
        if self._slots != NULL:
            free(self._slots)
            self._slots = NULL

    def __repr__(self) -> str:
        cdef Py_ssize_t size
        self._lock()
        try:
            size = self._size
        finally:
            self._unlock()
        return f"Cache(maxsize={self._maxsize}, currsize={size})"

    def __len__(self) -> int:
        cdef Py_ssize_t size
        self._lock()
        try:
            size = self._size
        finally:
            self._unlock()
        return size

    def __contains__(self, object key) -> bool:
        cdef bint found
        self._lock()
        try:
            found = self._lookup(key, NULL) != NULL
        finally:
            self._unlock()
        return found

    def __iter__(self):
        cdef list keys
        cdef Py_ssize_t i
        cdef Node* node
        self._lock()
        try:
            keys = []
            for i in range(self._nslots):
                node = self._slots[i].node
                if node != NULL and node != _TOMB:
                    keys.append(<object>node.key)
        finally:
            self._unlock()
        return iter(keys)

    cdef inline Node* _lookup_hashed(self, object key, uint64_t khash) except? NULL:
        cdef Py_ssize_t i = <Py_ssize_t>(khash & <uint64_t>self._slot_mask)
        cdef Py_ssize_t n
        cdef int cmp
        cdef Node* node
        for n in range(self._nslots):
            node = self._slots[i].node
            if node == NULL:
                return NULL
            if node != _TOMB and self._slots[i].khash == khash:
                if node.key == <PyObject*>key:
                    return node
                cmp = PyObject_RichCompareBool(<object>node.key, key, Py_EQ)
                if cmp == -1:
                    raise
                if cmp == 1:
                    return node
            i = (i + 1) & self._slot_mask
        return NULL

    cdef inline Node* _lookup(self, object key, uint64_t* khash_out) except? NULL:
        cdef uint64_t khash = <uint64_t><Py_hash_t>hash(key)
        if khash_out != NULL:
            khash_out[0] = khash
        return self._lookup_hashed(key, khash)

    cdef inline Node* _lookup2(self, object a, object b, uint64_t khash) except? NULL:
        cdef Py_ssize_t i = <Py_ssize_t>(khash & <uint64_t>self._slot_mask)
        cdef Py_ssize_t n
        cdef Node* node
        for n in range(self._nslots):
            node = self._slots[i].node
            if node == NULL:
                return NULL
            if (
                node != _TOMB
                and self._slots[i].khash == khash
                and _tuple2_match(<object>node.key, a, b)
            ):
                return node
            i = (i + 1) & self._slot_mask
        return NULL

    cdef inline Node* _lookup3(
        self, object a, object b, object c, uint64_t khash
    ) except? NULL:
        cdef Py_ssize_t i = <Py_ssize_t>(khash & <uint64_t>self._slot_mask)
        cdef Py_ssize_t n
        cdef Node* node
        for n in range(self._nslots):
            node = self._slots[i].node
            if node == NULL:
                return NULL
            if (
                node != _TOMB
                and self._slots[i].khash == khash
                and _tuple3_match(<object>node.key, a, b, c)
            ):
                return node
            i = (i + 1) & self._slot_mask
        return NULL

    cdef void _place(self, Node* node) except *:
        cdef Py_ssize_t i = <Py_ssize_t>(node.khash & <uint64_t>self._slot_mask)
        cdef Py_ssize_t n
        cdef Node* cur
        for n in range(self._nslots):
            cur = self._slots[i].node
            if cur == NULL or cur == _TOMB:
                if cur == _TOMB:
                    self._tombs -= 1
                self._slots[i].khash = node.khash
                self._slots[i].node = node
                node.slot = i
                self._size += 1
                return
            i = (i + 1) & self._slot_mask
        self._rehash()
        self._place(node)

    cdef void _rehash(self) except *:
        cdef Slot* old = self._slots
        cdef Py_ssize_t old_n = self._nslots
        cdef Py_ssize_t i
        cdef Node* node
        self._slots = <Slot*>calloc(<size_t>self._nslots, sizeof(Slot))
        if self._slots == NULL:
            self._slots = old
            raise MemoryError()
        self._size = 0
        self._tombs = 0
        for i in range(old_n):
            node = old[i].node
            if node != NULL and node != _TOMB:
                self._place(node)
        free(old)

    cdef void _drop(self, Node* node) except *:
        cdef bint compact = False
        if node == NULL:
            return
        self._slots[node.slot].node = _TOMB
        self._size -= 1
        self._tombs += 1
        compact = self._tombs * 2 >= self._nslots
        if node.key != NULL and self._keep is not None:
            self._keep.pop(<object>node.key, None)
        self._node_release(node)
        if compact:
            self._rehash()

    cdef void _discard_nodes(self):
        cdef Py_ssize_t i
        cdef Py_ssize_t count
        cdef Node* node
        cdef Node** pending
        if self._slots == NULL:
            return
        pending = <Node**>malloc(<size_t>self._nslots * sizeof(Node*))
        count = 0
        for i in range(self._nslots):
            node = self._slots[i].node
            if node != NULL and node != _TOMB:
                if pending != NULL:
                    pending[count] = node
                    count += 1
                else:
                    self._node_release(node)
        memset(self._slots, 0, <size_t>self._nslots * sizeof(Slot))
        self._size = 0
        self._tombs = 0
        if self._keep is not None:
            self._keep.clear()
        if self._window_s != NULL:
            self._window_s.prev = self._window_s
            self._window_s.next = self._window_s
        if self._probation_s != NULL:
            self._probation_s.prev = self._probation_s
            self._probation_s.next = self._probation_s
        if self._protected_s != NULL:
            self._protected_s.prev = self._protected_s
            self._protected_s.next = self._protected_s
        self._window_n = 0
        self._probation_n = 0
        self._protected_n = 0
        if pending != NULL:
            for i in range(count):
                self._node_release(pending[i])
            free(pending)

    cdef inline object _probe(self, object key, object default, bint fail):
        cdef uint64_t khash
        cdef Node* node = self._lookup(key, &khash)
        if node == NULL:
            self._misses += 1
            self._sketch.increment_hash(khash)
            if fail:
                raise KeyError(key)
            return default
        self._hits += 1
        self._on_hit(node)
        return <object>node.value

    cdef void _insert_new(self, object key, object value, uint64_t khash) except *:
        cdef Node* node = self._node_acquire(key, value, khash)
        self._sketch.increment_hash(khash)
        self._keep[key] = value
        self._place(node)
        _push(self._window_s, node, WINDOW, &self._window_n)
        self._evict_window()

    def __getitem__(self, object key):
        self._lock()
        try:
            return self._probe(key, None, True)
        finally:
            self._unlock()

    def __setitem__(self, object key, object value):
        cdef uint64_t khash
        cdef Node* node
        self._lock()
        try:
            node = self._lookup(key, &khash)
            if node != NULL:
                node.value = <PyObject*>value
                self._keep[key] = value
                self._on_hit(node)
                return
            self._insert_new(key, value, khash)
        finally:
            self._unlock()

    cdef void _evict_node(self, Node* node) except *:
        if node.region == WINDOW:
            _remove(node, &self._window_n)
        elif node.region == PROBATION:
            _remove(node, &self._probation_n)
        else:
            _remove(node, &self._protected_n)
        self._drop(node)

    def __delitem__(self, object key):
        cdef Node* node
        self._lock()
        try:
            node = self._lookup(key, NULL)
            if node == NULL:
                raise KeyError(key)
            self._evict_node(node)
        finally:
            self._unlock()

    cpdef object pop(self, object key, object default=_MISSING):
        cdef Node* node
        cdef object value
        self._lock()
        try:
            node = self._lookup(key, NULL)
            if node == NULL:
                if default is _MISSING:
                    raise KeyError(key)
                return default
            value = <object>node.value
            self._evict_node(node)
            return value
        finally:
            self._unlock()

    cpdef object setdefault(self, object key, object default=None):
        cdef uint64_t khash
        cdef Node* node
        self._lock()
        try:
            node = self._lookup(key, &khash)
            if node != NULL:
                self._hits += 1
                self._on_hit(node)
                return <object>node.value
            self._insert_new(key, default, khash)
            return default
        finally:
            self._unlock()

    cpdef object get(self, object key, object default=None):
        self._lock()
        try:
            return self._probe(key, default, False)
        finally:
            self._unlock()

    cpdef void clear(self):
        self._lock()
        try:
            self._discard_nodes()
            self._sketch = FrequencySketch(self._maxsize)
            self._hits = 0
            self._misses = 0
        finally:
            self._unlock()

    cpdef int _frequency(self, object key):
        """Count-Min estimate for tests (0-15). Not part of the public API."""
        cdef int freq
        self._lock()
        try:
            freq = self._sketch.frequency(key)
        finally:
            self._unlock()
        return freq

    cdef inline object _memo_get(self, object key, uint64_t* khash_out):
        cdef Node* node
        self._lock()
        try:
            node = self._lookup(key, khash_out)
            if node != NULL:
                self._hits += 1
                self._on_hit(node)
                return <object>node.value
            self._misses += 1
        finally:
            self._unlock()
        return _MISSING

    cdef inline object _memo_get2(self, object a, object b, uint64_t khash):
        cdef Node* node
        self._lock()
        try:
            node = self._lookup2(a, b, khash)
            if node != NULL:
                self._hits += 1
                self._on_hit(node)
                return <object>node.value
            self._misses += 1
        finally:
            self._unlock()
        return _MISSING

    cdef inline object _memo_get3(self, object a, object b, object c, uint64_t khash):
        cdef Node* node
        self._lock()
        try:
            node = self._lookup3(a, b, c, khash)
            if node != NULL:
                self._hits += 1
                self._on_hit(node)
                return <object>node.value
            self._misses += 1
        finally:
            self._unlock()
        return _MISSING

    cdef object _memo_store(self, object key, object value, uint64_t khash):
        cdef Node* node
        self._lock()
        try:
            node = self._lookup_hashed(key, khash)
            if node != NULL:
                return <object>node.value
            self._insert_new(key, value, khash)
            return value
        finally:
            self._unlock()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def hits(self) -> int:
        cdef Py_ssize_t n
        self._lock()
        try:
            n = self._hits
        finally:
            self._unlock()
        return n

    @property
    def misses(self) -> int:
        cdef Py_ssize_t n
        self._lock()
        try:
            n = self._misses
        finally:
            self._unlock()
        return n

    cdef inline void _on_hit(self, Node* node) noexcept:
        self._sketch.increment_hash(node.khash)
        if node.region == WINDOW:
            if node.next != self._window_s:
                _unlink(node)
                _link_mru(self._window_s, node)
        elif node.region == PROBATION:
            _remove(node, &self._probation_n)
            _push(self._protected_s, node, PROTECTED, &self._protected_n)
            self._demote_protected()
        elif node.next != self._protected_s:
            _unlink(node)
            _link_mru(self._protected_s, node)

    cdef inline void _demote_protected(self) noexcept:
        cdef Node* node
        while self._protected_n > self._protected_max:
            node = _pop_lru(self._protected_s, &self._protected_n)
            if node == NULL:
                break
            _push(self._probation_s, node, PROBATION, &self._probation_n)

    cdef void _evict_window(self) except *:
        cdef Node* candidate
        while self._window_n > self._window_max:
            candidate = _pop_lru(self._window_s, &self._window_n)
            if candidate == NULL:
                break
            self._admit(candidate)

    cdef void _admit(self, Node* candidate) except *:
        cdef Node* victim
        if candidate == NULL:
            return
        if self._probation_n + self._protected_n < self._main_max:
            _push(self._probation_s, candidate, PROBATION, &self._probation_n)
            return
        victim = _peek_lru(self._probation_s)
        if victim == NULL:
            self._drop(candidate)
            return
        if self._sketch.frequency_hash(candidate.khash) > self._sketch.frequency_hash(
            victim.khash
        ):
            victim = _pop_lru(self._probation_s, &self._probation_n)
            if victim == NULL:
                self._drop(candidate)
                return
            self._drop(victim)
            _push(self._probation_s, candidate, PROBATION, &self._probation_n)
        else:
            self._drop(candidate)

    cdef Py_ssize_t _walk_list(self, Node* sentinel) except -1:
        cdef Py_ssize_t n = 0
        cdef Node* node
        if sentinel == NULL:
            return 0
        node = sentinel.next
        while node != sentinel:
            n += 1
            if n > self._maxsize + 8:
                raise RuntimeError("region list is corrupt")
            if node == NULL or node.prev == NULL or node.next == NULL:
                raise RuntimeError("region list has a NULL link")
            node = node.next
        return n

    cpdef dict _debug_counts(self):
        """Internal sizes for tests."""
        cdef dict counts
        self._lock()
        try:
            counts = self._debug_counts_unlocked()
        finally:
            self._unlock()
        return counts

    cdef dict _debug_counts_unlocked(self):
        cdef Py_ssize_t live = 0
        cdef Py_ssize_t tombs = 0
        cdef Py_ssize_t i
        cdef Node* node
        cdef Py_ssize_t window_walk
        cdef Py_ssize_t probation_walk
        cdef Py_ssize_t protected_walk
        for i in range(self._nslots):
            node = self._slots[i].node
            if node == NULL:
                continue
            if node == _TOMB:
                tombs += 1
                continue
            live += 1
            if node.slot != i:
                raise RuntimeError("slot index mismatch")
            if node.key == NULL or node.value == NULL:
                raise RuntimeError("live node lost a Python object")
        window_walk = self._walk_list(self._window_s)
        probation_walk = self._walk_list(self._probation_s)
        protected_walk = self._walk_list(self._protected_s)
        if live != self._size:
            raise RuntimeError("live slots != size")
        if tombs != self._tombs:
            raise RuntimeError("tomb count mismatch")
        if window_walk != self._window_n:
            raise RuntimeError("window count mismatch")
        if probation_walk != self._probation_n:
            raise RuntimeError("probation count mismatch")
        if protected_walk != self._protected_n:
            raise RuntimeError("protected count mismatch")
        if self._window_n + self._probation_n + self._protected_n != self._size:
            raise RuntimeError("region sizes != size")
        if self._keep is None or len(self._keep) != self._size:
            raise RuntimeError("GC keep set != size")
        if self._size > self._maxsize:
            raise RuntimeError("cache grew past maxsize")
        if self._window_n > self._window_max:
            raise RuntimeError("window grew past window_max")
        if self._protected_n > self._protected_max:
            raise RuntimeError("protected grew past protected_max")
        return {
            "size": self._size,
            "window": self._window_n,
            "probation": self._probation_n,
            "protected": self._protected_n,
            "tombs": self._tombs,
            "nslots": self._nslots,
            "window_max": self._window_max,
            "main_max": self._main_max,
            "protected_max": self._protected_max,
            "sketch_samples": self._sketch.samples,
            "sketch_sample_size": self._sketch.sample_size,
        }


CacheInfo = namedtuple("CacheInfo", ("hits", "misses", "maxsize", "currsize"))


cdef object _make_key(tuple args, dict kwargs, bint typed):
    cdef object key
    if kwargs:
        key = args + tuple(sorted(kwargs.items()))
    else:
        key = args
    if typed:
        if kwargs:
            return (
                key,
                tuple([type(a) for a in args]),
                tuple([(k, type(v)) for k, v in sorted(kwargs.items())]),
            )
        return (key, tuple([type(a) for a in args]))
    return key


cdef object _cache_info(Cache cache):
    cache._lock()
    try:
        return CacheInfo(cache._hits, cache._misses, cache._maxsize, cache._size)
    finally:
        cache._unlock()


cdef class _Memoized:
    cdef Cache cache
    cdef bint typed
    cdef public object __wrapped__
    cdef public object __name__
    cdef public object __doc__
    cdef object _fn_module

    def __cinit__(self, fn, Py_ssize_t maxsize, bint typed):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.cache = Cache(maxsize)
        self.typed = typed
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, "__name__", "memoized")
        self.__doc__ = getattr(fn, "__doc__", None)
        self._fn_module = getattr(fn, "__module__", "")

    @property
    def __module__(self):
        return self._fn_module

    def __call__(self, *args, **kwargs):
        cdef uint64_t khash
        cdef object key = _make_key(args, kwargs, self.typed)
        cdef object value = self.cache._memo_get(key, &khash)
        if value is not _MISSING:
            return value
        value = self.__wrapped__(*args, **kwargs)
        return self.cache._memo_store(key, value, khash)

    def cache_info(self):
        return _cache_info(self.cache)

    def cache_clear(self):
        self.cache.clear()

    def cache_parameters(self) -> dict:
        return {"maxsize": self.cache.maxsize, "typed": self.typed}


cdef class _MemoizedOne:
    """One positional argument: no ``*args`` tuple on the hit path."""

    cdef Cache cache
    cdef public object __wrapped__
    cdef public object __name__
    cdef public object __doc__
    cdef object _fn_module

    def __cinit__(self, fn, Py_ssize_t maxsize):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.cache = Cache(maxsize)
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, "__name__", "memoized")
        self.__doc__ = getattr(fn, "__doc__", None)
        self._fn_module = getattr(fn, "__module__", "")

    @property
    def __module__(self):
        return self._fn_module

    def __call__(self, arg):
        cdef uint64_t khash
        cdef object value = self.cache._memo_get(arg, &khash)
        if value is not _MISSING:
            return value
        value = self.__wrapped__(arg)
        return self.cache._memo_store(arg, value, khash)

    def cache_info(self):
        return _cache_info(self.cache)

    def cache_clear(self):
        self.cache.clear()

    def cache_parameters(self) -> dict:
        return {"maxsize": self.cache.maxsize, "typed": False}


cdef class _MemoizedTwo:
    """Two positional arguments: no ``*args`` tuple on the hit path."""

    cdef Cache cache
    cdef public object __wrapped__
    cdef public object __name__
    cdef public object __doc__
    cdef object _fn_module

    def __cinit__(self, fn, Py_ssize_t maxsize):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.cache = Cache(maxsize)
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, "__name__", "memoized")
        self.__doc__ = getattr(fn, "__doc__", None)
        self._fn_module = getattr(fn, "__module__", "")

    @property
    def __module__(self):
        return self._fn_module

    def __call__(self, a, b):
        cdef uint64_t khash = _hash2(a, b)
        cdef object value = self.cache._memo_get2(a, b, khash)
        if value is not _MISSING:
            return value
        value = self.__wrapped__(a, b)
        return self.cache._memo_store((a, b), value, khash)

    def cache_info(self):
        return _cache_info(self.cache)

    def cache_clear(self):
        self.cache.clear()

    def cache_parameters(self) -> dict:
        return {"maxsize": self.cache.maxsize, "typed": False}


cdef class _MemoizedThree:
    """Three positional arguments: no ``*args`` tuple on the hit path."""

    cdef Cache cache
    cdef public object __wrapped__
    cdef public object __name__
    cdef public object __doc__
    cdef object _fn_module

    def __cinit__(self, fn, Py_ssize_t maxsize):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.cache = Cache(maxsize)
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, "__name__", "memoized")
        self.__doc__ = getattr(fn, "__doc__", None)
        self._fn_module = getattr(fn, "__module__", "")

    @property
    def __module__(self):
        return self._fn_module

    def __call__(self, a, b, c):
        cdef uint64_t khash = _hash3(a, b, c)
        cdef object value = self.cache._memo_get3(a, b, c, khash)
        if value is not _MISSING:
            return value
        value = self.__wrapped__(a, b, c)
        return self.cache._memo_store((a, b, c), value, khash)

    def cache_info(self):
        return _cache_info(self.cache)

    def cache_clear(self):
        self.cache.clear()

    def cache_parameters(self) -> dict:
        return {"maxsize": self.cache.maxsize, "typed": False}


def memoize(maxsize: int = 128, typed: bool = False):
    """Memoize ``fn`` with a Window-TinyLFU cache of ``maxsize`` entries."""

    def deco(fn):
        code = getattr(fn, "__code__", None)
        if (
            not typed
            and code is not None
            and code.co_kwonlyargcount == 0
            and getattr(fn, "__defaults__", None) is None
            and not code.co_flags & 0x0C
        ):
            if code.co_argcount == 1:
                return _MemoizedOne(fn, maxsize)
            if code.co_argcount == 2:
                return _MemoizedTwo(fn, maxsize)
            if code.co_argcount == 3:
                return _MemoizedThree(fn, maxsize)
        return _Memoized(fn, maxsize, typed)

    return deco
