// ===================== OPTIMIZED VP-TREE (fixed) ============================
// • Matrix rows/cols tracked correctly
// • embed_ / real_ members added
// • Heap uses std::priority_queue.size() instead of sz
// • AVX loads in cosine() use unaligned (_mm256_loadu_ps)
// • build() & search() compile with C++20 and AVX2
// ===========================================================================

#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <immintrin.h>
#include <utility>
#include <type_traits>
#include <memory>
#include <queue>
#include <stdbool.h>
#include <fstream>

// ─── quick pow(q) helpers (unchanged) ────────────────────────────────────────
constexpr float pow_q_fast(float x, float q) {
    if (q == 1.f)  return x;
    if (q == 3.f)  return x * x * x;
    if (q == 5.f)  return x * x * x * x * x;
    if (q == 10.f) { float x2=x*x,x4=x2*x2; return x4*x4*x2; }
    if (q == 15.f) { float x3=x*x*x,x5=x3*x*x; return x5*x5*x*x*x; }
    if (q == 20.f) { float x2=x*x,x4=x2*x2,x8=x4*x4; return x8*x8*x4; }
    if (q == 30.f) { float x2=x*x,x3=x2*x,x6=x3*x3,x15=x6*x6*x3; return x15*x15; }
    return std::pow(x,q);
}

/* ---- Matrix  ---------------------------------------------------------- */
struct Matrix {
    int dim = 0;
    std::vector<float> data;

    void from_vecvec(const std::vector<std::vector<float>>& src) {
        dim = src.empty() ? 0 : int(src[0].size());
        data.reserve(src.size() * dim);
        for (auto& row : src) data.insert(data.end(), row.begin(), row.end());
    }
    const float* row_ptr(int r) const { return data.data() + r * dim; }
    inline int rows() const { return dim == 0 ? 0 : int(data.size() / dim); }
};

// ─── VP-Tree ----------------------------------------------------------------
class VpTree {
public:

    enum class Metric {
        Euclidean,
        Manhattan,
        Cosine,
        Jaccard,
        Custom = -1
    };
    struct SearchResult { std::vector<int> ids, dists; };

    VpTree() = default;

    VpTree(float q, Metric metric_embed, Metric metric_real, float eps)
        : q_(q)
        , eps_(eps)
        , metric_embed_(metric_embed)
        , metric_real_(metric_real)
        , rng_(std::random_device{}())
    {
        if (q_ <= 0.f) throw std::invalid_argument("q must be > 0");
    }


    void create(std::vector<std::vector<float>> real,
                std::vector<std::vector<float>> embed,
                std::vector<int> items = {});

    SearchResult search(int k,
                        const std::vector<float>& qE,
                        const std::vector<float>& qR,
                        bool retDist = false) const {
        return search_core(k, qE.data(), qR.data(), retDist);
    }

    std::vector<std::vector<int>>
    search_batch(int k, int topk,
                 const float* qE, const float* qR,
                 std::size_t nQuery,
                 bool retDist=false) const;

    void set_custom_real(std::function<double(const float*, const float*, int)> f) {
        custom_real_fn_ = std::move(f);
    }

    void set_custom_embed(std::function<double(const float*, const float*, int)> f) {
        custom_embed_fn_ = std::move(f);
    }


    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open file");

        int version = 1;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&q_), sizeof(q_));
        out.write(reinterpret_cast<const char*>(&eps_), sizeof(eps_));

        int metE = static_cast<int>(metric_embed_);
        int metR = static_cast<int>(metric_real_);
        out.write(reinterpret_cast<const char*>(&metE), sizeof(int));
        out.write(reinterpret_cast<const char*>(&metR), sizeof(int));

        int dimE = embed_.dim, rowsE = embed_.rows();
        out.write(reinterpret_cast<const char*>(&dimE), sizeof(dimE));
        out.write(reinterpret_cast<const char*>(&rowsE), sizeof(rowsE));
        out.write(reinterpret_cast<const char*>(embed_.data.data()), embed_.data.size() * sizeof(float));

        int dimR = real_.dim, rowsR = real_.rows();
        out.write(reinterpret_cast<const char*>(&dimR), sizeof(dimR));
        out.write(reinterpret_cast<const char*>(&rowsR), sizeof(rowsR));
        out.write(reinterpret_cast<const char*>(real_.data.data()), real_.data.size() * sizeof(float));

        int item_count = int(items_.size());
        out.write(reinterpret_cast<const char*>(&item_count), sizeof(item_count));
        out.write(reinterpret_cast<const char*>(items_.data()), item_count * sizeof(int));

        int node_count = int(nodes_.size());
        out.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
        out.write(reinterpret_cast<const char*>(nodes_.data()), node_count * sizeof(Node));

        out.write(reinterpret_cast<const char*>(&root_), sizeof(root_));

        int norm_size = int(real_norms_.size());
        out.write(reinterpret_cast<const char*>(&norm_size), sizeof(norm_size));
        out.write(reinterpret_cast<const char*>(real_norms_.data()), norm_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(embed_norms_.data()), norm_size * sizeof(double));
    }


    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open file");

        int version;
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1)
            throw std::runtime_error("Unsupported index file version");

        in.read(reinterpret_cast<char*>(&q_), sizeof(q_));
        in.read(reinterpret_cast<char*>(&eps_), sizeof(eps_));

        int metE, metR;
        in.read(reinterpret_cast<char*>(&metE), sizeof(metE));
        in.read(reinterpret_cast<char*>(&metR), sizeof(metR));
        metric_embed_ = static_cast<Metric>(metE);
        metric_real_  = static_cast<Metric>(metR);

        // Load embed_ matrix
        int dimE, rowsE;
        in.read(reinterpret_cast<char*>(&dimE), sizeof(dimE));
        in.read(reinterpret_cast<char*>(&rowsE), sizeof(rowsE));
        embed_.dim = dimE;
        embed_.data.resize(dimE * rowsE);
        in.read(reinterpret_cast<char*>(embed_.data.data()), embed_.data.size() * sizeof(float));

        // Load real_ matrix
        int dimR, rowsR;
        in.read(reinterpret_cast<char*>(&dimR), sizeof(dimR));
        in.read(reinterpret_cast<char*>(&rowsR), sizeof(rowsR));
        real_.dim = dimR;
        real_.data.resize(dimR * rowsR);
        in.read(reinterpret_cast<char*>(real_.data.data()), real_.data.size() * sizeof(float));

        // Load items
        int item_count;
        in.read(reinterpret_cast<char*>(&item_count), sizeof(item_count));
        items_.resize(item_count);
        in.read(reinterpret_cast<char*>(items_.data()), item_count * sizeof(int));

        // Load nodes
        int node_count;
        in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
        nodes_.resize(node_count);
        in.read(reinterpret_cast<char*>(nodes_.data()), node_count * sizeof(Node));

        // Load root index
        in.read(reinterpret_cast<char*>(&root_), sizeof(root_));

        // Load norms
        int norm_size;
        in.read(reinterpret_cast<char*>(&norm_size), sizeof(norm_size));
        real_norms_.resize(norm_size);
        embed_norms_.resize(norm_size);
        in.read(reinterpret_cast<char*>(real_norms_.data()), norm_size * sizeof(double));
        in.read(reinterpret_cast<char*>(embed_norms_.data()), norm_size * sizeof(double));

        // Rebuild rows_
        int n = embed_.rows();
        rows_.resize(n);
        std::iota(rows_.begin(), rows_.end(), 0);
    }




private:
    double l2(const float* a,const float* b,int d) const;
    double l1(const float* a,const float* b,int d) const;
    double cosine(const float* a,const float* b,int d,double na,double nb) const;
    double jaccard(const float* a,const float* b,int d) const;
    double dist_embed(const float* a,const float* b,int row) const;
    double dist_real (const float* a,const float* b,int row) const;
    std::function<double(const float*, const float*, int)> custom_real_fn_ = nullptr;
    std::function<double(const float*, const float*, int)> custom_embed_fn_ = nullptr;

    struct Heap {
        using Elem = std::pair<double,int>;
        int k;
        std::priority_queue<Elem> pq;

        explicit Heap(int kk): k(kk) {
            if (k <= 0) throw std::invalid_argument("k must be > 0");
        }

        double worst() const {
            if ((int)pq.size() < k)
                return std::numeric_limits<double>::infinity();
            return pq.top().first;
        }

        void push(double dist, int row) {
            if ((int)pq.size() < k) {
                pq.emplace(dist, row);
            } else if (dist < pq.top().first) {
                pq.pop();
                pq.emplace(dist, row);
            }
        }

        void collect(std::vector<int>& out) const {
            out.clear();
            std::vector<Elem> tmp;
            tmp.reserve(pq.size());
            auto copy = pq;
            while (!copy.empty()) {
                tmp.push_back(copy.top());
                copy.pop();
            }
            std::reverse(tmp.begin(), tmp.end());
            out.reserve(tmp.size());
            for (auto &e : tmp)
                out.push_back(e.second);
        }
    };

    struct Node {
        int   row   = 0;
        float thr   = 0.0f;
        __float128 thr_q = 0.0;
        int   left  = -1;
        int   right = -1;
    };

    int  build(int lo,int hi);
    void search_rec(int node,int k,Heap& h) const;
    SearchResult search_core(int k,const float* qE,const float* qR,bool retDist) const;

    Matrix embed_, real_;
    std::vector<double> embed_norms_, real_norms_;
    float q_, eps_;
    Metric metric_embed_, metric_real_;
    std::mt19937 rng_;
    std::vector<Node> nodes_;
    std::vector<int> rows_, items_;
    int root_ = -1;

    mutable const float* qE_ptr_ = nullptr;
    mutable const float* qR_ptr_ = nullptr;
    mutable double embed_norm_q_ = 0, real_norm_q_ = 0;
    mutable float tau_ = std::numeric_limits<float>::infinity();
};

// ── kernels ───────────────────────────────────────────────────────────────
inline double VpTree::l2(const float* a,const float* b,int d) const {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i+8 <= d; i += 8) {
        auto va = _mm256_loadu_ps(a+i);
        auto vb = _mm256_loadu_ps(b+i);
        auto diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    float buf[8]; _mm256_storeu_ps(buf, acc);
    double s = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    for (; i < d; ++i) s += double(a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(s);
}
inline double VpTree::l1(const float* a,const float* b,int d) const {
    __m256 acc = _mm256_setzero_ps();
    const __m256 neg = _mm256_set1_ps(-0.f);
    int i = 0;
    for (; i+8 <= d; i += 8) {
        auto diff = _mm256_sub_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i));
        auto absp = _mm256_andnot_ps(neg, diff);
        acc = _mm256_add_ps(acc, absp);
    }
    float buf[8]; _mm256_storeu_ps(buf, acc);
    double s = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    for (; i < d; ++i) s += std::fabs(a[i] - b[i]);
    return s;
}
inline double VpTree::cosine(const float* a,const float* b,int d,double na,double nb) const {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i+8 <= d; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    }
    float buf[8]; _mm256_storeu_ps(buf, acc);
    double dot = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    for (; i < d; ++i) dot += static_cast<double>(a[i]) * b[i];
    return (na==0||nb==0) ? 1.0 : 1.0 - dot / (std::sqrt(na) * std::sqrt(nb));
}
inline double VpTree::jaccard(const float* a,const float* b,int d) const {
    __m256 sumI = _mm256_setzero_ps();
    __m256 sumU = _mm256_setzero_ps();
    int i = 0;
    for (; i+8 <= d; i += 8) {
        auto va = _mm256_loadu_ps(a+i);
        auto vb = _mm256_loadu_ps(b+i);
        sumI = _mm256_add_ps(sumI, _mm256_mul_ps(va, vb));
        sumU = _mm256_add_ps(sumU, _mm256_max_ps(va, vb));
    }
    float bufI[8], bufU[8];
    _mm256_storeu_ps(bufI, sumI);
    _mm256_storeu_ps(bufU, sumU);
    double inter=0, uni=0;
    for (int j=0; j<8; ++j) { inter += bufI[j]; uni += bufU[j]; }
    for (; i<d; ++i) { inter += double(a[i] * b[i]); uni += double(std::max(a[i], b[i])); }
    return (uni==0) ? 1.0 : 1.0 - inter / uni;
}

inline double VpTree::dist_embed(const float* a,const float* b,int r) const {
    switch (metric_embed_) {
        case Metric::Euclidean: return l2(a, b, embed_.dim);
        case Metric::Manhattan: return l1(a, b, embed_.dim);
        case Metric::Cosine:    return cosine(a, b, embed_.dim, embed_norm_q_, embed_norms_[r]);
        case Metric::Jaccard:   return jaccard(a, b, embed_.dim);
        case Metric::Custom:
            if (custom_embed_fn_) return custom_embed_fn_(a, b, r);
    }
    return 0.0;
}
inline double VpTree::dist_real(const float* a,const float* b,int r) const {
    switch (metric_real_) {
        case Metric::Euclidean: return l2(a, b, real_.dim);
        case Metric::Manhattan: return l1(a, b, real_.dim);
        case Metric::Cosine:    return cosine(a, b, real_.dim, real_norm_q_, real_norms_[r]);
        case Metric::Jaccard:   return jaccard(a, b, real_.dim);
        case Metric::Custom:
            if (custom_real_fn_) return custom_real_fn_(a, b, r);
    }
    return 0.0;
}

inline void VpTree::create(std::vector<std::vector<float>> real,
                           std::vector<std::vector<float>> embed,
                           std::vector<int> items) {
    if (real.empty() || real.size() != embed.size())
        throw std::invalid_argument("real & embed size mismatch");
    real_.from_vecvec(real);
    embed_.from_vecvec(embed);
    int n = embed_.rows();
    if (!items.empty()) {
        if ((int)items.size() != n)
            throw std::invalid_argument("items length mismatch");
        items_ = std::move(items);
    } else {
        items_.resize(n);
        std::iota(items_.begin(), items_.end(), 0);
    }
    rows_.resize(n);
    std::iota(rows_.begin(), rows_.end(), 0);

    if (metric_embed_ == Metric::Cosine || metric_real_ == Metric::Cosine) {
        embed_norms_.resize(n);
        real_norms_.resize(n);
        for (int i = 0; i < n; ++i) {
            double se=0, sr=0;
            const float* pe = embed_.row_ptr(i);
            const float* pr = real_.row_ptr(i);
            for (int j = 0; j < embed_.dim; ++j) se += double(pe[j])*pe[j];
            for (int j = 0; j < real_.dim;  ++j) sr += double(pr[j])*pr[j];
            embed_norms_[i] = se;
            real_norms_[i]  = sr;
        }
    }

    nodes_.clear(); nodes_.reserve(n);
    root_ = build(0, n);
}

inline int VpTree::build(int lo,int hi) {
    if (hi <= lo) return -1;
    int cur = (int)nodes_.size();
    nodes_.emplace_back();
    std::swap(rows_[lo], rows_[std::uniform_int_distribution<int>(lo,hi-1)(rng_)]);
    Node node;
    int pivot = rows_[lo];
    node.row = pivot;
    int cnt = hi - lo - 1;
    if (cnt > 0) {
        std::vector<std::pair<float,int>> tmp(cnt);
        const float* p = embed_.row_ptr(node.row);
        for (int j=0; j<cnt; ++j) {
            int r = rows_[lo+1+j];
            tmp[j] = {float(dist_embed(p, embed_.row_ptr(r), r)), r};
        }
        std::sort(tmp.begin(), tmp.end(), [](auto &a, auto &b){return a.first<b.first;});
        for (int j=0; j<cnt; ++j) rows_[lo+1+j] = tmp[j].second;

        int m = cnt/2;
        node.thr   = tmp[m].first;
        if (std::isinf(q_)) {
            // for the ∞‐case, thr_q should be µ_v itself,
            // so we can use it directly in pruning
            node.thr_q = node.thr;
        }
        else {
            node.thr_q = pow_q_fast(node.thr, q_);
        }

        node.left  = build(lo+1, lo+1+m);
        node.right = build(lo+1+m, hi);
    }
    nodes_[cur] = node;
    return cur;
}

inline void VpTree::search_rec(int idx,int k,Heap& h) const {
    if (idx<0) return;
    const Node &nd = nodes_[idx];
    double d = dist_embed(qE_ptr_, embed_.row_ptr(nd.row), nd.row);
    if (h.pq.size()<size_t(k) || d < h.worst()) {
        h.push(d, nd.row);
        tau_ = float(h.worst());
    }
    if (nd.left<0 && nd.right<0) return;

    // 4) compute pruning flags
    double dist_q, tau_q, thr_q;
    bool check_left, check_right;

    if (isinf(q_)) {
        // q = ∞ case
        dist_q = d;
        thr_q  = nd.thr;    // μ_v
        tau_q  = tau_;        // τ

        if (fmax(dist_q, tau_q) <= thr_q) {
            check_left  = true;  check_right = false;
        }
        else if (dist_q >= fmax(thr_q, tau_q)) {
            check_left  = false; check_right = true;
        }
        else {
            check_left  = true;  check_right = true;
        }
    }
    else {
        // finite-q case

        dist_q = pow(d,q_);
        thr_q  = nd.thr_q;
        tau_q  = pow(tau_,q_);

        if (dist_q + tau_q <= thr_q) {
            check_left  = true;  check_right = false;
        }
        else if (dist_q >= thr_q + tau_q) {
            check_left  = false; check_right = true;
        }
        else {
            check_left  = true;  check_right = true;
        }
    }

    // 5) recurse nearer-first

    if (dist_q < thr_q) {
        if (check_left)  search_rec(nd.left,  k, h);
        if (check_right) search_rec( nd.right, k, h);
    } else {
        if (check_right) search_rec( nd.right, k, h);
        if (check_left)  search_rec( nd.left,  k, h);
    }
}

inline VpTree::SearchResult VpTree::search_core(int k,const float* qE,const float* qR,bool retDist) const {
    SearchResult res;
    if (k<=0 || root_<0) return res;
    qE_ptr_=qE; qR_ptr_=qR;
    embed_norm_q_=real_norm_q_=0;
    if (metric_embed_==Metric::Cosine) for(int i=0;i<embed_.dim;++i) embed_norm_q_+=double(qE[i])*qE[i];
    if (metric_real_==Metric::Cosine)  for(int i=0;i<real_.dim;++i)  real_norm_q_ +=double(qR[i])*qR[i];
    Heap h(k); tau_=std::numeric_limits<float>::infinity();
    search_rec(root_,k,h);
    std::vector<int> cand;
    h.collect(cand);
    if (cand.empty()) return res;
    std::vector<std::pair<double,int>> buf;
    buf.reserve(cand.size());
    for(int r:cand) buf.emplace_back(dist_real(real_.row_ptr(r),qR_ptr_,r),r);
    int m=int(buf.size()), top=std::min(k,m);
    std::partial_sort(buf.begin(), buf.begin()+top, buf.end(),[](auto&a,auto&b){return a.first<b.first;});
    res.ids.resize(top);
    if (retDist) res.dists.resize(top);
    for(int i=0;i<top;++i) {
        res.ids[i]=items_[buf[i].second];
        if(retDist) res.dists[i]=float(buf[i].first);
    }
    return res;
}

inline std::vector<std::vector<int>>
VpTree::search_batch(int k,int topk,const float* qE,const float* qR,std::size_t N,bool retDist) const {
    std::vector<std::vector<int>> all(N);
    int dE=embed_.dim, dR=real_.dim;
    for(size_t i=0;i<N;++i) {
        auto r = search_core(k, qE+i*dE, qR+i*dR, retDist);
        if ((int)r.ids.size() > topk)
            r.ids.resize(topk);
        all[i] = std::move(r.ids);
    }
    return all;
}

