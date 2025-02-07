#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <chrono>

double linearInterpolate(const std::vector<double> &xdat, const std::vector<double> &ydat,
                         double x);

template <int DIM, typename FUNC>
void forNeighbors(const std::array<int, DIM> &centerIdx, const std::array<int, DIM> &range,
                  FUNC &&callback);

template <int DIM>
double distancePeriodic(const std::array<double, DIM> &a, const std::array<double, DIM> &b,
                        const std::array<double, DIM> &length, bool periodic);

template <int DIM>
struct Cell {
    std::vector<std::vector<std::array<double, DIM>>> coords;
    std::vector<std::vector<double>> deathRates;
    std::vector<int> population;
    std::vector<double> cellBirthRateBySpecies;
    std::vector<double> cellDeathRateBySpecies;
    double cellBirthRate = 0.0;
    double cellDeathRate = 0.0;

    Cell() = default;

    void initSpecies(int M) {
        coords.resize(M);
        deathRates.resize(M);
        population.resize(M, 0);
        cellBirthRateBySpecies.resize(M, 0.0);
        cellDeathRateBySpecies.resize(M, 0.0);
    }
};

template <int DIM>
class Grid {
public:
    int M;
    std::array<double, DIM> area_length;
    std::array<int, DIM> cell_count;
    bool periodic;
    std::vector<double> b;
    std::vector<double> d;
    std::vector<std::vector<double>> dd;
    std::vector<std::vector<double>> birth_x;
    std::vector<std::vector<double>> birth_y;
    std::vector<std::vector<std::vector<double>>> death_x;
    std::vector<std::vector<std::vector<double>>> death_y;
    std::vector<std::vector<double>> cutoff;
    std::vector<std::vector<std::array<int, DIM>>> cull;
    std::vector<Cell<DIM>> cells;
    int total_num_cells;
    double total_birth_rate{0.0};
    double total_death_rate{0.0};
    int total_population{0};
    std::mt19937 rng;
    double time{0.0};
    int event_count{0};
    std::chrono::system_clock::time_point init_time;
    double realtime_limit;
    bool realtime_limit_reached{false};

    Grid(int M_, const std::array<double, DIM> &areaLen, const std::array<int, DIM> &cellCount_,
         bool isPeriodic, const std::vector<double> &birthRates,
         const std::vector<double> &deathRates, const std::vector<double> &ddMatrix,
         const std::vector<std::vector<double>> &birthX,
         const std::vector<std::vector<double>> &birthY,
         const std::vector<std::vector<std::vector<double>>> &deathX_,
         const std::vector<std::vector<std::vector<double>>> &deathY_,
         const std::vector<double> &cutoffs, int seed, double rtimeLimit);

    int flattenIdx(const std::array<int, DIM> &idx) const;
    std::array<int, DIM> unflattenIdx(int cellIndex) const;
    int wrapIndex(int i, int dim) const;
    bool inDomain(const std::array<int, DIM> &idx) const;
    Cell<DIM> &cellAt(const std::array<int, DIM> &raw);
    double evalBirthKernel(int s, double x) const;
    double evalDeathKernel(int s1, int s2, double dist) const;
    std::array<double, DIM> randomUnitVector(std::mt19937 &rng);
    void spawn_at(int s, const std::array<double, DIM> &inPos);
    void kill_at(int s, const std::array<int, DIM> &cIdx, int victimIdx);
    void removeInteractionsOfParticle(const std::array<int, DIM> &cIdx, int sVictim, int victimIdx);
    void placePopulation(const std::vector<std::vector<std::array<double, DIM>>> &initCoords);
    void spawn_random();
    void kill_random();
    void make_event();
    void run_events(int events);
    void run_for(double duration);
    std::vector<std::vector<std::array<double, DIM>>> get_all_particle_coords() const;
    std::vector<std::vector<double>> get_all_particle_death_rates() const;
};

extern template class Grid<1>;
extern template class Grid<2>;
extern template class Grid<3>;

extern template struct Cell<1>;
extern template struct Cell<2>;
extern template struct Cell<3>;
