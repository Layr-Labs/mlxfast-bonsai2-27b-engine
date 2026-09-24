import Foundation

/// Compressed exact-token index. Values exist only at explicitly inserted
/// endpoints: splitting an edge never invents a reusable recurrent checkpoint.
/// The owner supplies synchronization and removes values when storage retires.
struct CBv2TokenRadixIndex<Value: Hashable> {
    private final class Node {
        var edge: [Int]
        var values: Set<Value> = []
        var children: [Int: Node] = [:]

        init(edge: [Int]) { self.edge = edge }
    }

    private let root = Node(edge: [])

    mutating func insert(tokens: ArraySlice<Int>, value: Value) {
        guard !tokens.isEmpty else { return }
        var node = root
        var remaining = tokens
        while let first = remaining.first {
            guard let child = node.children[first] else {
                let leaf = Node(edge: Array(remaining))
                leaf.values.insert(value)
                node.children[first] = leaf
                return
            }
            let shared = Self.commonPrefix(child.edge, remaining)
            if shared < child.edge.count {
                let branch = Node(edge: Array(child.edge.prefix(shared)))
                child.edge.removeFirst(shared)
                branch.children[child.edge[0]] = child
                node.children[first] = branch
                node = branch
            } else {
                node = child
            }
            remaining = remaining.dropFirst(shared)
        }
        node.values.insert(value)
    }

    /// Endpoints in increasing token order, limited to the supplied prefix.
    /// No hashing, allocation proportional to cache size, or token approximation.
    func matches(tokens: ArraySlice<Int>) -> [(position: Int, values: Set<Value>)] {
        var node = root
        var remaining = tokens
        var position = 0
        var result: [(position: Int, values: Set<Value>)] = []
        while let first = remaining.first, let child = node.children[first] {
            guard Self.commonPrefix(child.edge, remaining) == child.edge.count else { break }
            position += child.edge.count
            remaining = remaining.dropFirst(child.edge.count)
            node = child
            if !node.values.isEmpty { result.append((position, node.values)) }
        }
        return result
    }

    mutating func remove(tokens: ArraySlice<Int>, value: Value) {
        var node = root
        var remaining = tokens
        var path: [(parent: Node, key: Int)] = []
        while let first = remaining.first, let child = node.children[first] {
            guard Self.commonPrefix(child.edge, remaining) == child.edge.count else { return }
            path.append((node, first))
            remaining = remaining.dropFirst(child.edge.count)
            node = child
        }
        guard remaining.isEmpty, node.values.remove(value) != nil else { return }
        for (parent, key) in path.reversed() {
            guard let child = parent.children[key], child.values.isEmpty else { break }
            if child.children.isEmpty {
                parent.children.removeValue(forKey: key)
            } else if child.children.count == 1, let grandchild = child.children.values.first {
                child.edge.append(contentsOf: grandchild.edge)
                child.values = grandchild.values
                child.children = grandchild.children
            }
        }
    }

    var isEmpty: Bool { root.children.isEmpty }

    private static func commonPrefix(_ edge: [Int], _ tokens: ArraySlice<Int>) -> Int {
        var matched = 0
        for (lhs, rhs) in zip(edge, tokens) {
            guard lhs == rhs else { break }
            matched += 1
        }
        return matched
    }
}
