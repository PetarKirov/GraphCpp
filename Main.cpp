//Compile with -std=c++1y
#include <cassert>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <initializer_list>
#include <vector>
#include <stack>
#include <queue>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <limits>
#include <type_traits>

template<typename T>
using Seq = std::initializer_list<T>;

template<typename T>
using Set = std::set<T>;

template <typename T>
T& top(std::stack<T>& s) { return s.top(); }

template <typename T>
T& top(std::queue<T>& q) { return q.front(); }

//Square matrix containing n^2 elements
//template<class T>
class Matrix
{
public:
	//typedef T value_type;
	typedef int value_type;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef value_type* iterator;
	typedef const iterator const_iterator;
	typedef size_t size_type;

	// Construct empty matrix
	Matrix() noexcept { }

	// Allocate memory for n^2 elements, n should be > 1. Uninitialized.
	Matrix(size_type n_)
		: n{ n_ }, data{ new value_type[n_ * n_] }
	{
		assert(n_ > 1);

		std::fill(begin(), end(), value_type{});
	}

	// Construct from literal, e.g. Matrix{ 0, 1, 2, 3 }, where rows==cols==2
	Matrix(Seq<value_type> l)
	{
		// assert that l.size() is a perfect square
		// and also prevent users trying to use Matrix(size_type)
		// from mistakenly calling Matrix{ size_type }
		size_type N = (size_type)floor(sqrt(l.size()));
		assert(double(N) == sqrt(l.size()) && N > 1);

		this->data = new value_type[N * N];
		this->n = N;

		std::copy(l.begin(), l.end(), begin());
	}

	Matrix(Matrix const& other) { this->copy(other); }

	// Strong exception-safety:
	// If allocation fails, leave object unchanged.
	Matrix& operator= (Matrix const& other)
	{
		if (this != &other)
			this->copy(other);

		return *this;
	}

	~Matrix() { free(); }

	reference operator() (size_type row, size_type col) { return get(row, col); }

	const_reference operator() (size_type row, size_type col) const { return get(row, col); }

	// Returns true if no data is currently allocated.
	// Returns false otherwise.
	bool empty() noexcept { return !(data && n); }

	// Returns the size of underlying 1D array, i.e. n^2
	size_type size() const noexcept { return n * n; }

	size_type rows() const noexcept { return this->n; }
	size_type cols() const noexcept { return this->n; }
	iterator begin() noexcept { return data; }
	iterator end() noexcept { return data + size(); }
	const_iterator begin() const noexcept { return data; }
	const_iterator end() const noexcept { return data + size(); }

	// Frees the allocated data.
	// Safe to call multiple (consecutive) times.
	void free() noexcept { free_impl(); }

private:
	size_type n = 0;
	value_type* data = nullptr;

	reference get(size_type row, size_type col) { return data[n * row + col]; }

	const_reference get(size_type row, size_type col) const { return data[n * row + col]; }

	// Frees the allocated data.
	// Safe to call multiple (consecutive) times.
	void free_impl() noexcept
	{
		delete[] data;
		data = nullptr;
		n = 0;
	}

	void copy(const Matrix& other)
	{
		if (this->n != other.n)
		{
			// Strong exception-safety:
			// If allocation fails, leave object unchanged.
			value_type* newData = new value_type[other.n * other.n];

			// If we are heare, the allocation must have succeeded.
			delete[] this->data;
			this->data = newData;
			this->n = other.n;
		}

		for (size_type i = 0; i != n * n; i++)
			this->data[i] = other.data[i];
	}
};

void testMatrix()
{
	// Test empty matrix constructor
	Matrix m{};
	assert(m.empty());
	assert(m.size() == 0);
	assert(m.rows() == m.cols() && m.rows() == 0);

	// Test 5 x 5 matrix
	Matrix m5x5(5);
	assert(!m5x5.empty());
	assert(m5x5.size() == 25);
	assert(m5x5.rows() == m5x5.cols() && m5x5.rows() == 5);

	// Test iterator access
	std::fill(m5x5.begin(), m5x5.end(), 3);
	for (const int& val : m5x5)
		assert(val == 3);

	// test operator(row, col)
	for (size_t i = 0; i != m5x5.rows(); i++)
		for (size_t j = 0; j != m5x5.cols(); j++)
			m5x5(i, j) = 10 * (i + 1) + (j + 1);

	for (size_t i = 0; i != m5x5.rows(); i++)
		for (size_t j = 0; j != m5x5.cols(); j++)
			assert((size_t)m5x5(i, j) == 10 * (i + 1) + (j + 1));

	// test std <algorithm>
	int n = -1;
	std::generate(m5x5.begin(), m5x5.end(),
		[&]() { n++; return n / m5x5.cols() * 10 + n % m5x5.cols(); });

	for (size_t i = 0; i != m5x5.rows(); i++)
		for (size_t j = 0; j != m5x5.cols(); j++)
			assert((size_t)m5x5(i, j) == 10 * i + j);

	// Test sequence constructor
	auto l = { 0, 1, 2, 3 };
	Matrix m2x2{ 0, 1, 2, 3 };
	assert(std::equal(l.begin(), l.end(),
		m2x2.begin(), m2x2.end()));
}


//eg Graph<char> or Graph<int>
template<class T, bool directed>
class Graph
{
public:
	using M = Matrix;
	using size_type = typename M::size_type;
	using edge_value_type = typename M::value_type;
	using Vertex = T;

	using Edge = struct E_ {
		const Vertex& i;
		const Vertex& j;
		bool operator<(const E_& lhs) const;
	};

	using WeightedEdge = struct WE_ {
		const Vertex& i;
		const Vertex& j;
		edge_value_type weight;
		bool operator<(const WE_& lhs) const;
	};

	using WeightFunc = std::function<edge_value_type(Edge)>;
	using Path = std::vector<Vertex>;
	using IndexPath = std::vector<size_type>;

	// Constructors taking std::initializer_list
	Graph(Seq<Vertex> vertices_, Seq<WeightedEdge> weightedEdges_)
	{
		fillVertices(vertices_.begin(), vertices_.end(), false);
		fillWeights(weightedEdges_.begin(), weightedEdges_.end());
	}

	Graph(Seq<Vertex> vertices_, Seq<Edge> edges_, WeightFunc func)
	{
		fillVertices(vertices_.begin(), vertices_.end(), false);
		fillWeights(edges_.begin(), edges_.end(), func);
	}

	// Constructors taking std::set
	Graph(const Set<Vertex>& vertices_, const Set<WeightedEdge>& weightedEdges_)
	{
		fillVertices(vertices_.begin(), vertices_.end(), true);
		fillWeights(weightedEdges_.begin(), weightedEdges_.end());
	}

	Graph(const Set<Vertex>& vertices_, const Set<Edge>& edges_, WeightFunc func)
	{
		fillVertices(vertices_.begin(), vertices_.end(), true);
		fillWeights(edges_.begin(), edges_.end(), func);
	}

	// Get internal representation
	const Matrix& getAdjacencyMatrix() const { return A; }

	const std::vector<Vertex>& getVertices() const { return vertices; }

	// Translate Vertex <-> size_type
	size_type getIndex(const Vertex& v) const
	{
		return std::distance(vertices.begin(),
			std::lower_bound(vertices.begin(), vertices.end(), v));
	}

	const Vertex& getVertex(size_type idx) const
	{
		return vertices[idx];
	}

	// Get/Set weight for particular edge
	int& operator[](Edge e)
	{
		return A(indexOf(e.first),
			indexOf(e.second));
	}

	const int& operator[](Edge e) const
	{
		return A(indexOf(e.first),
			indexOf(e.second));
	}

	IndexPath toIndexPath(Path path)
	{
		using namespace std;
		auto result = vector<size_type>(path.size());
		transform(path.begin(), path.end(), result.begin(),
			[this](const Vertex& v) { return getIndex(v); });

		return result;
	}

	void print_neighbours() const
	{
		using namespace std;

		for_each_vertex([this](const Vertex& v, size_type idx)
		{
			cout << endl << v << ": ";
			for_each_neighbour(idx, [this](size_type vidx, size_type j)
			{
				cout << endl << "-> " << getVertex(j) << ": " << A(vidx, j) << "km";
			});
		});
	}

	void for_each_neighbour(size_type vindex, std::function<void(size_type, size_type)> f) const
	{
		for (size_t j = 0; j < A.cols(); j++)
			if (A(vindex, j) > 0)
				f(vindex, j);
	}

	void for_each_vertex(std::function<void(const Vertex&, size_type)> f) const
	{
		for (size_t i = 0; i != vertices.size() - 1; i++)
			f(vertices[i], i);
	}

private:
	Matrix A; // Adjacency matrix
	std::vector<Vertex> vertices;

	template<typename Iterator>
	void fillVertices(Iterator begin, Iterator end, bool assumeSortedAndUnique)
	{
		using namespace std;
		vertices = vector<Vertex>(begin, end);
		if (!assumeSortedAndUnique)
		{
			sort(vertices.begin(), vertices.end());
			auto last = unique(vertices.begin(), vertices.end());
			vertices.resize(distance(vertices.begin(), last));
		}
		A = Matrix(vertices.size());
	}

	template<typename Iterator>
	void fillWeights(Iterator begin, Iterator end)
	{
		std::for_each(begin, end, [this](const WeightedEdge& we)
		{
			size_type i = getIndex(we.i);
			size_type j = getIndex(we.j);
			size_type weight = we.weight;

			A(i, j) = weight;
			if (!directed)
				A(j, i) = weight;
		});
	}

	template<typename Iterator>
	void fillWeights(Iterator begin, Iterator end, WeightFunc w)
	{
		std::for_each(begin, end, [this, &w](const Edge& e)
		{
			size_type i = getIndex(e.i);
			size_type j = getIndex(e.j);

			A(i, j) = w(e);
			if (!directed)
				A(j, i) = w(e);
		});
	}
};

template<class T, bool d>
bool Graph<T, d>::Edge::operator<(const Edge& lhs) const
{
	return i < lhs.i ? true :
		j < lhs.j;
}

template<class T, bool d>
bool Graph<T, d>::WeightedEdge::operator<(const WeightedEdge& lhs) const
{
	return i < lhs.i ? true :
		j < lhs.j ? true :
		weight < lhs.weight;
}

template<typename T, bool directed>
struct Traverser
{
	using G = Graph<T, directed>;
	using M = typename G::M;
	using size_type = typename Graph<T, directed>::size_type;
	using edge_value_type = typename G::edge_value_type;
	using Vertex = typename G::Vertex;
	using Path = typename G::Path;
	using IndexPath = typename G::IndexPath;
	using Depth = struct { const Vertex& vertex; size_type level; };
	using DepthI = struct { size_type idx; size_type level; };
	using VisitFunc = std::function<void(Depth)>;
	enum class Traversal { DepthFirst, BreadthFirst };

	Traverser(const G& graph_)
		: graph(graph_), A(graph.getAdjacencyMatrix()), visited(A.rows(), false) { }

	void BFS(Vertex start, VisitFunc visitor)
	{
		traverse<Traversal::BreadthFirst>(graph.getIndex(start), visitor);
	}

	void DFS(Vertex start, VisitFunc visitor)
	{
		traverse<Traversal::DepthFirst>(graph.getIndex(start), visitor);
	}


	IndexPath shortestPath(Vertex start, Vertex end)
	{
		const auto s = graph.getIndex(start);
		const auto e = graph.getIndex(end);
		const auto inf_dist = std::numeric_limits<edge_value_type>::max();
		const auto null_parent = std::numeric_limits<size_type>::max();
		struct VD {
			size_type vertex;
			edge_value_type distance;
			bool operator< (const VD& other) const
			{
				return distance < other.distance;
			}
		};

		auto queue = std::priority_queue<VD>();
		auto dist = std::vector<edge_value_type>(A.rows());
		auto path = IndexPath(A.rows());

		for (size_type i = 0; i != A.rows(); ++i)
		{
			dist[i] = inf_dist;
			path[i] = null_parent;
		}

		dist[s] = 0;
		queue.push({ s, dist[s] });

		while (!queue.empty())
		{
			auto closest = queue.top(); queue.pop();

			if (closest.distance == inf_dist) break;

			graph.for_each_neighbour(closest.vertex, [closest, this, &queue](size_type c, size_type j)
			{
				VD neighbor = { j, A(closest.vertex, j) };
				auto potDistance = closest.distance + neighbor.distance;
				if (potDistance < neighbor.distance)
				{
					neighbor.distance = potDistance;
					queue.push(neighbor);
				}
			});
		}


		return path;
	}

private:
	const G& graph;
	const M& A;
	std::vector<bool> visited;

	template<Traversal traversal>
	void traverse(size_type start, VisitFunc visitor)
	{
		using Cont = std::conditional_t<
			traversal == Traversal::BreadthFirst,
			std::queue<DepthI>,
			std::stack<DepthI >> ;

		auto cont = Cont{};
		visited = std::vector<bool>(A.rows());

		cont.push({ start, 0 });

		while (!cont.empty())
		{
			auto v = top(cont);
			cont.pop();

			if (!visited[v.idx]) {
				visitor({ graph.getVertex(v.idx), v.level });
				visited[v.idx] = true;
			}

			for (size_type col = 0; col != A.cols(); ++col)
				if (A(v.idx, col) && !visited[col])
					cont.push({ col, v.level + 1 });
		}
	}
};

template<typename T, bool directed>
Traverser<T, directed> get(const Graph<T, directed>& graph)
{
	return Traverser<T, directed>(graph);
}

void testGraph()
{
	using namespace std;
	typedef Graph<char, true> G1;
	typedef Graph<string, false> G2;

	Set<G1::Vertex> vertices{ 'a', 'b', 'c' };
	Set<G1::WeightedEdge> weightedEdges =
	{
		{ 'a', 'b', 2 },
		{ 'a', 'c', 3 }
	};

	G1 g1(vertices, weightedEdges);

	G2::WeightFunc w = [](const G2::Edge& e) { return 2; };

	G2 g2(
	{
		"Sofia"s, "Pernik"s, "Vraza"s, "Lovech"s,
		"Pleven"s, "Pazardjik"s, "Plovdiv"s, "Stara Zagora"s,
		"Burgas"s, "Varna"s, "Dobrich"s, "Shumen"s,
		"Razgrad"s, "Ruse"s, "Targovishte"s, "Veliko Tarnovo"s,
		"Gabrovo"s,

	},
	{
		{ "Sofia"s, "Pernik"s, 34 },
		{ "Sofia"s, "Vraza"s, 111 },
		{ "Sofia"s, "Lovech"s, 152 },
		{ "Sofia"s, "Pazardjik"s, 112 },
		{ "Lovech"s, "Pleven"s, 39 },
		{ "Pazardjik"s, "Plovdiv"s, 39 },
		{ "Plovdiv"s, "Stara Zagora"s, 102 },
		{ "Stara Zagora"s, "Burgas"s, 172 },
		{ "Burgas"s, "Varna"s, 129 },
		{ "Varna"s, "Dobrich"s, 54 },
		{ "Varna"s, "Shumen"s, 89 },
		{ "Dobrich"s, "Razgrad"s, 138 },
		{ "Shumen"s, "Razgrad"s, 50 },
		{ "Shumen"s, "Targovishte"s, 41 },
		{ "Razgrad"s, "Targovishte"s, 37 },
		{ "Razgrad"s, "Ruse"s, 65 },
		{ "Ruse"s, "Pleven"s, 152 },
		{ "Targovishte"s, "Veliko Tarnovo"s, 100 },
		{ "Veliko Tarnovo"s, "Gabrovo"s, 45 },
		{ "Veliko Tarnovo"s, "Lovech"s, 84 },
	});

		auto p = g2.toIndexPath({ "Ruse"s, "Sofia"s, "Burgas"s, "Pleven"s, "Varna"s });

		for (const auto& i : p)
			cout << i << " ";

		cout << endl;

		auto t = get(g2);
		t.BFS("Sofia"s,
			[](decltype(t)::Depth d)
		{
			cout << string(d.level, ' ') << d.level << ": " << d.vertex << endl;
		});

		cout << endl;

		t.DFS("Sofia"s,
			[](decltype(t)::Depth d)
		{
			cout << string(d.level, ' ') << d.level << ": " << d.vertex << endl;
		});

		g2.print_neighbours();

		cout << endl;

		auto path = t.shortestPath("Sofia", "Dobrich");

		for (auto x : path)
			cout << x << " ";
}




int main()
{
	testMatrix();
	testGraph();
}

#if 0

struct pairhash;

template<typename T>
struct IsPair;

template<typename E>
using Set = typename std::conditional_t<
	IsPair<E>::value,
	std::set<E>, pairhash>,
	std::set<E>
			>;

struct pairhash
{
public:
	template <typename T, typename U>
	std::size_t operator()(const std::pair<T, U> &x) const
	{
		return std::hash<T>()(x.first) ^
			std::hash<U>()(x.second);
	}
};

template<typename T>
struct IsPair
{
private:
	template<typename C> static std::true_type test(typename C::second_type*);
	template<typename C> static std::false_type test(...);
public:
	static const bool value = decltype(test<T>(0))::value;//sizeof(test<T>(0)) == sizeof(yes);
};

#endif
