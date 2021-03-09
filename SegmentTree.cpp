#include <assert.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <variant>
#include <vector>

using std::vector;

constexpr int16_t SEGMENT_TREE_NODE_DEFAULT_VALUE = -20'000;

template <typename T>
const T& operation_on_segment(const T& a, const T& b) {
    return std::max<T>(a, b);
}

struct Segment {
    size_t first;
    size_t last;

    size_t size() const {
        return end() - begin();
    }

    size_t end() const {
        return last + 1;
    }

    size_t begin() const {
        return first;
    }

    Segment operator+(const size_t segment_shift) const {
        Segment result{ first + segment_shift, last + segment_shift };
        return result;
    }
};

enum class SIDE { LEFT, RIGHT };

template <size_t dimension_number>
class Point {
public:
    size_t position;
    Point<dimension_number - 1> less_dimension;

    explicit Point(const vector<size_t>& coordinates)
        : position(coordinates[dimension_number - 1]),
        less_dimension(coordinates) {}
};

template <>
class Point<0> {
public:
    explicit Point(const vector<size_t>& coordinates) {}
};

template <size_t dimension_number>
class Space {
public:
    Segment segment;
    Space<dimension_number - 1> less_dimension;

    explicit Space(const vector<Segment>& coordinates)
        : segment(coordinates[dimension_number - 1]),
        less_dimension(coordinates) {}
};

template <>
class Space<0> {
public:
    explicit Space(const vector<Segment>& coordinates) {}
};

template <size_t dimension_number>
class SegmentTree {
public:
    explicit SegmentTree(const Space<dimension_number>& parameters)
        : nodes(make_nodes(parameters)) {}

    void change_value(const Point<dimension_number>& point,
        const int16_t new_value) {
        size_t index = to_absolute(point.position);

        nodes[index].change_value(point.less_dimension, new_value);
        while (!is_root(index)) {
            index = parent_index(index);
            update_node(index, point.less_dimension);
        }
    }

    int16_t value_in_space(const Space<dimension_number>& space) const {
        size_t left = to_absolute(space.segment.first);
        size_t right = to_absolute(space.segment.last);

        const int16_t left_corner_value =
            nodes[left].value_in_space(space.less_dimension);
        const int16_t right_corner_value =
            nodes[right].value_in_space(space.less_dimension);
        int16_t result =
            operation_on_segment(left_corner_value, right_corner_value);

        while (parent_index(left) != parent_index(right)) {
            const int16_t temporary_result =
                middle_brothers_values(left, right, space.less_dimension);

            result = operation_on_segment(result, temporary_result);

            left = parent_index(left);
            right = parent_index(right);
        }
        return result;
    }

    int16_t value_at_point(const Point<dimension_number>& point) const {
        const size_t index = to_absolute(point.position);

        const auto result = nodes[index].value_at_point(point.less_dimension);
        return result;
    }

private:
    void update_node(const size_t index,
        const Point<dimension_number - 1>& point) {
        const int16_t left_son_value = child_value<SIDE::LEFT>(index, point);
        const int16_t right_son_value = child_value<SIDE::RIGHT>(index, point);
        const int16_t new_value =
            operation_on_segment(left_son_value, right_son_value);

        nodes[index].change_value(point, new_value);
    }

    template <SIDE side>
    size_t child_index(const size_t parent) const {
        const size_t is_right_child = side == SIDE::RIGHT;
        const size_t index_of_child = parent * 2 + is_right_child;
        if (index_of_child >= nodes.size()) {
            throw std::out_of_range("Leaves don't have children");
        }
        return index_of_child;
    }

    template <SIDE side>
    int16_t child_value(const size_t parent,
        const Point<dimension_number - 1>& point) const {
        const size_t index_of_child = child_index<side>(parent);

        const int16_t result = nodes[index_of_child].value_at_point(point);
        return result;
    }

    template <SIDE side>
    int16_t child_value(const size_t parent,
        const Space<dimension_number - 1>& space) const {
        const size_t index_of_child = child_index<side>(parent);

        const int16_t result = nodes[index_of_child].value_in_space(space);
        return result;
    }

    template <SIDE brother_side>
    int16_t brother_value(const size_t child,
        const Space<dimension_number - 1>& space) const {
        const SIDE child_side = as_child_side(child);
        if (child_side != brother_side) {
            const size_t parent = parent_index(child);
            const int16_t brother_value = child_value<brother_side>(parent,
                space);
            return brother_value;
        }
        else {
            return SEGMENT_TREE_NODE_DEFAULT_VALUE;
        }
    }

    int16_t middle_brothers_values(
        const size_t left_child, const size_t right_child,
        const Space<dimension_number - 1>& space) const {
        const int16_t right_brother_of_left_child =
            brother_value<SIDE::RIGHT>(left_child, space);
        const int16_t left_brother_of_right_child =
            brother_value<SIDE::LEFT>(right_child, space);

        const auto result = operation_on_segment(right_brother_of_left_child,
            left_brother_of_right_child);
        return result;
    }

    size_t to_absolute(const size_t index) const {
        const size_t absolute_index = index + nodes.size() / 2;
        if (absolute_index >= nodes.size()) {
            throw std::out_of_range("Point is out tree range");
        }
        return absolute_index;
    }

    size_t parent_index(const size_t child) const {
        if (is_root(child)) {
            throw std::out_of_range("Root doesn't have parent");
        }
        return child / 2;
    }

    SIDE as_child_side(const size_t child) const {
        if (child % 2 == 1) {
            return SIDE::RIGHT;
        }
        else {
            return SIDE::LEFT;
        }
    }

    bool is_root(const size_t index) const noexcept {
        return index == root_index;
    }

    static size_t size_required_for_tree(const size_t leaves_count) {
        const size_t upper_power_of_2 = log2(leaves_count - 1) + 2;

        return 1 << upper_power_of_2;
    }

    static vector<SegmentTree<dimension_number - 1>> make_nodes(
        const Space<dimension_number>& parameters) {
        const size_t nodes_count =
            size_required_for_tree(parameters.segment.size());
        const auto default_node =
            SegmentTree<dimension_number - 1>(parameters.less_dimension);

        vector<SegmentTree<dimension_number - 1>> result(nodes_count,
            default_node);
        return result;
    }

    vector<SegmentTree<dimension_number - 1>> nodes;
    static constexpr size_t root_index = 1;
};

template <>
class SegmentTree<0> {
public:
    explicit SegmentTree(const Space<0> parameters)
        : value(SEGMENT_TREE_NODE_DEFAULT_VALUE) {}

    void change_value(const Point<0> point, const int16_t value_) {
        value = value_;
    }

    int16_t value_in_space(const Space<0> space) const {
        return value;
    }

    int16_t value_at_point(const Point<0> point) const {
        return value;
    }

private:
    int16_t value;
};