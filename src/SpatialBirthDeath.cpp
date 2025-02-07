#include <vector>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <random>
#include "../include/SpatialBirthDeath.h"

double linearInterpolate(const std::vector<double> &xdat, const std::vector<double> &ydat,
                         double x) {
    if (x >= xdat.back()) {
        return ydat.back();
    }
    if (x <= xdat.front()) {
        return ydat.front();
    }
    auto i = std::lower_bound(xdat.begin(), xdat.end(), x);
    const size_t k = i - xdat.begin();
    const size_t l = (k > 0) ? k - 1 : 0;
    const double x1 = xdat[l];
    const double x2 = xdat[k];
    const double y1 = ydat[l];
    const double y2 = ydat[k];
    return y1 + ((y2 - y1) * (x - x1) / (x2 - x1));
}

template <int DIM>
double distancePeriodic(const std::array<double, DIM> &a, const std::array<double, DIM> &b,
                        const std::array<double, DIM> &length, bool periodic) {
    double sumSq = 0.0;
    for (int i = 0; i < DIM; i++) {
        double diff = a[i] - b[i];
        if (periodic) {
            if (diff > 0.5 * length[i]) {
                diff -= length[i];
            } else if (diff < -0.5 * length[i]) {
                diff += length[i];
            }
        }
        sumSq += diff * diff;
    }
    return std::sqrt(sumSq);
}

template <int DIM, typename FUNC>
void forNeighbors(const std::array<int, DIM> &centerIdx, const std::array<int, DIM> &range,
                  FUNC &&callback) {
    std::array<int, DIM> neighborIdx;
    if constexpr (DIM == 1) {
        const int minX = centerIdx[0] - range[0];
        const int maxX = centerIdx[0] + range[0];
        for (int x = minX; x <= maxX; x++) {
            neighborIdx[0] = x;
            callback(neighborIdx);
        }
    } else if constexpr (DIM == 2) {
        const int minX = centerIdx[0] - range[0];
        const int maxX = centerIdx[0] + range[0];
        const int minY = centerIdx[1] - range[1];
        const int maxY = centerIdx[1] + range[1];
        for (int x = minX; x <= maxX; x++) {
            neighborIdx[0] = x;
            for (int y = minY; y <= maxY; y++) {
                neighborIdx[1] = y;
                callback(neighborIdx);
            }
        }
    } else if constexpr (DIM == 3) {
        const int minX = centerIdx[0] - range[0];
        const int maxX = centerIdx[0] + range[0];
        const int minY = centerIdx[1] - range[1];
        const int maxY = centerIdx[1] + range[1];
        const int minZ = centerIdx[2] - range[2];
        const int maxZ = centerIdx[2] + range[2];
        for (int x = minX; x <= maxX; x++) {
            neighborIdx[0] = x;
            for (int y = minY; y <= maxY; y++) {
                neighborIdx[1] = y;
                for (int z = minZ; z <= maxZ; z++) {
                    neighborIdx[2] = z;
                    callback(neighborIdx);
                }
            }
        }
    }
}

template <int DIM>
Grid<DIM>::Grid(int M_, const std::array<double, DIM> &areaLen,
                const std::array<int, DIM> &cellCount_, bool isPeriodic,
                const std::vector<double> &birthRates, const std::vector<double> &deathRates,
                const std::vector<double> &ddMatrix, const std::vector<std::vector<double>> &birthX,
                const std::vector<std::vector<double>> &birthY,
                const std::vector<std::vector<std::vector<double>>> &deathX_,
                const std::vector<std::vector<std::vector<double>>> &deathY_,
                const std::vector<double> &cutoffs, int seed, double rtimeLimit)
    : M(M_),
      area_length(areaLen),
      cell_count(cellCount_),
      periodic(isPeriodic),
      rng(seed),
      realtime_limit(rtimeLimit) {
    init_time = std::chrono::system_clock::now();
    b = birthRates;
    d = deathRates;
    dd.resize(M);
    for (int s1 = 0; s1 < M; s1++) {
        dd[s1].resize(M);
        for (int s2 = 0; s2 < M; s2++) {
            dd[s1][s2] = ddMatrix[(s1 * M) + s2];
        }
    }
    birth_x = birthX;
    birth_y = birthY;
    death_x.resize(M);
    death_y.resize(M);
    for (int s1 = 0; s1 < M; s1++) {
        death_x[s1].resize(M);
        death_y[s1].resize(M);
        for (int s2 = 0; s2 < M; s2++) {
            death_x[s1][s2] = deathX_[s1][s2];
            death_y[s1][s2] = deathY_[s1][s2];
        }
    }
    cutoff.resize(M);
    for (int s1 = 0; s1 < M; s1++) {
        cutoff[s1].resize(M);
        for (int s2 = 0; s2 < M; s2++) {
            cutoff[s1][s2] = cutoffs[(s1 * M) + s2];
        }
    }
    cull.resize(M);
    for (int s1 = 0; s1 < M; s1++) {
        cull[s1].resize(M);
        for (int s2 = 0; s2 < M; s2++) {
            for (int dim = 0; dim < DIM; dim++) {
                const double cellSize = area_length[dim] / cell_count[dim];
                const int needed = (int)std::ceil(cutoff[s1][s2] / cellSize);
                cull[s1][s2][dim] = std::max(needed, 3);
            }
        }
    }
    {
        int prod = 1;
        for (int dim = 0; dim < DIM; dim++) {
            prod *= cell_count[dim];
        }
        total_num_cells = prod;
    }
    cells.resize(total_num_cells);
    for (auto &c : cells) {
        c.initSpecies(M);
    }
}

template <int DIM>
int Grid<DIM>::flattenIdx(const std::array<int, DIM> &idx) const {
    int f = 0;
    int mul = 1;
    for (int dim = 0; dim < DIM; dim++) {
        f += idx[dim] * mul;
        mul *= cell_count[dim];
    }
    return f;
}

template <int DIM>
std::array<int, DIM> Grid<DIM>::unflattenIdx(int cellIndex) const {
    std::array<int, DIM> cIdx;
    for (int dim = 0; dim < DIM; dim++) {
        cIdx[dim] = cellIndex % cell_count[dim];
        cellIndex /= cell_count[dim];
    }
    return cIdx;
}

template <int DIM>
int Grid<DIM>::wrapIndex(int i, int dim) const {
    if (!periodic) {
        return i;
    }
    const int n = cell_count[dim];
    if (i < 0) {
        i += n;
    } else if (i >= n) {
        i -= n;
    }
    return i;
}

template <int DIM>
bool Grid<DIM>::inDomain(const std::array<int, DIM> &idx) const {
    for (int dim = 0; dim < DIM; dim++) {
        if (idx[dim] < 0 || idx[dim] >= cell_count[dim]) {
            return false;
        }
    }
    return true;
}

template <int DIM>
Cell<DIM> &Grid<DIM>::cellAt(const std::array<int, DIM> &raw) {
    std::array<int, DIM> w;
    for (int dim = 0; dim < DIM; dim++) {
        w[dim] = wrapIndex(raw[dim], dim);
    }
    return cells[flattenIdx(w)];
}

template <int DIM>
double Grid<DIM>::evalBirthKernel(int s, double x) const {
    return linearInterpolate(birth_x[s], birth_y[s], x);
}

template <int DIM>
double Grid<DIM>::evalDeathKernel(int s1, int s2, double dist) const {
    return linearInterpolate(death_x[s1][s2], death_y[s1][s2], dist);
}

template <int DIM>
std::array<double, DIM> Grid<DIM>::randomUnitVector(std::mt19937 &rng) {
    std::array<double, DIM> dir;
    if constexpr (DIM == 1) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        dir[0] = (u(rng) < 0.5) ? -1.0 : 1.0;
    } else {
        std::normal_distribution<double> gauss(0.0, 1.0);
        double sumSq = 0.0;
        for (int d = 0; d < DIM; d++) {
            const double val = gauss(rng);
            dir[d] = val;
            sumSq += val * val;
        }
        const double inv = 1.0 / std::sqrt(sumSq + 1e-14);
        for (int d = 0; d < DIM; d++) {
            dir[d] *= inv;
        }
    }
    return dir;
}

template <int DIM>
void Grid<DIM>::spawn_at(int s, const std::array<double, DIM> &inPos) {
    std::array<double, DIM> pos = inPos;
    for (int d = 0; d < DIM; d++) {
        if (pos[d] < 0.0 || pos[d] > area_length[d]) {
            if (!periodic) {
                return;
            }
            const double L = area_length[d];
            while (pos[d] < 0.0) {
                pos[d] += L;
            }
            while (pos[d] >= L) {
                pos[d] -= L;
            }
        }
    }
    std::array<int, DIM> cIdx;
    for (int d = 0; d < DIM; d++) {
        int c = (int)std::floor(pos[d] * cell_count[d] / area_length[d]);
        if (c == cell_count[d]) {
            c--;
        }
        cIdx[d] = c;
    }
    Cell<DIM> &cell = cellAt(cIdx);
    cell.coords[s].push_back(pos);
    cell.deathRates[s].push_back(d[s]);
    cell.population[s]++;
    total_population++;
    cell.cellBirthRateBySpecies[s] += b[s];
    cell.cellBirthRate += b[s];
    total_birth_rate += b[s];
    cell.cellDeathRateBySpecies[s] += d[s];
    cell.cellDeathRate += d[s];
    total_death_rate += d[s];
    auto &posNew = cell.coords[s].back();
    int newIdx = (int)cell.coords[s].size() - 1;
    for (int s2 = 0; s2 < M; s2++) {
        auto cullRange = cull[s][s2];
        forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int, DIM> &nIdx) {
            if (!periodic && !inDomain(nIdx)) {
                return;
            }
            Cell<DIM> &neighCell = cellAt(nIdx);
            for (int j = 0; j < (int)neighCell.coords[s2].size(); j++) {
                if (&neighCell == &cell && s2 == s && j == newIdx) {
                    continue;
                }
                auto &pos2 = neighCell.coords[s2][j];
                const double dist = distancePeriodic<DIM>(posNew, pos2, area_length, periodic);
                if (dist <= cutoff[s][s2]) {
                    const double inter_ij = dd[s][s2] * evalDeathKernel(s, s2, dist);
                    neighCell.deathRates[s2][j] += inter_ij;
                    neighCell.cellDeathRateBySpecies[s2] += inter_ij;
                    neighCell.cellDeathRate += inter_ij;
                    total_death_rate += inter_ij;
                }
                if (dist <= cutoff[s2][s]) {
                    const double inter_ji = dd[s2][s] * evalDeathKernel(s2, s, dist);
                    cell.deathRates[s][newIdx] += inter_ji;
                    cell.cellDeathRateBySpecies[s] += inter_ji;
                    cell.cellDeathRate += inter_ji;
                    total_death_rate += inter_ji;
                }
            }
        });
    }
}

template <int DIM>
void Grid<DIM>::kill_at(int s, const std::array<int, DIM> &cIdx, int victimIdx) {
    Cell<DIM> &cell = cellAt(cIdx);
    const double victimRate = cell.deathRates[s][victimIdx];
    cell.population[s]--;
    total_population--;
    cell.cellDeathRateBySpecies[s] -= victimRate;
    cell.cellDeathRate -= victimRate;
    total_death_rate -= victimRate;
    cell.cellBirthRateBySpecies[s] -= b[s];
    cell.cellBirthRate -= b[s];
    total_birth_rate -= b[s];
    removeInteractionsOfParticle(cIdx, s, victimIdx);
    const int lastIdx = (int)cell.coords[s].size() - 1;
    if (victimIdx != lastIdx) {
        cell.coords[s][victimIdx] = cell.coords[s][lastIdx];
        cell.deathRates[s][victimIdx] = cell.deathRates[s][lastIdx];
    }
    cell.coords[s].pop_back();
    cell.deathRates[s].pop_back();
}

template <int DIM>
void Grid<DIM>::removeInteractionsOfParticle(const std::array<int, DIM> &cIdx, int sVictim,
                                             int victimIdx) {
    Cell<DIM> &victimCell = cellAt(cIdx);
    auto &posVictim = victimCell.coords[sVictim][victimIdx];
    for (int s2 = 0; s2 < M; s2++) {
        auto range = cull[sVictim][s2];
        forNeighbors<DIM>(cIdx, range, [&](const std::array<int, DIM> &nIdx) {
            if (!periodic && !inDomain(nIdx)) {
                return;
            }
            Cell<DIM> &neighCell = cellAt(nIdx);
            for (int j = 0; j < (int)neighCell.coords[s2].size(); j++) {
                if (&neighCell == &victimCell && s2 == sVictim && j == victimIdx) {
                    continue;
                }
                auto &pos2 = neighCell.coords[s2][j];
                const double dist = distancePeriodic<DIM>(posVictim, pos2, area_length, periodic);
                if (dist <= cutoff[sVictim][s2]) {
                    const double inter_ij = dd[sVictim][s2] * evalDeathKernel(sVictim, s2, dist);
                    neighCell.deathRates[s2][j] -= inter_ij;
                    neighCell.cellDeathRateBySpecies[s2] -= inter_ij;
                    neighCell.cellDeathRate -= inter_ij;
                    total_death_rate -= inter_ij;
                }
            }
        });
    }
}

template <int DIM>
void Grid<DIM>::placePopulation(
    const std::vector<std::vector<std::array<double, DIM>>> &initCoords) {
    for (int s = 0; s < M; s++) {
        for (auto &pos : initCoords[s]) {
            spawn_at(s, pos);
        }
    }
}

template <int DIM>
void Grid<DIM>::spawn_random() {
    if (total_birth_rate < 1e-12) {
        return;
    }
    std::vector<double> cellRateVec(total_num_cells);
    for (int i = 0; i < total_num_cells; i++) {
        cellRateVec[i] = cells[i].cellBirthRate;
    }
    std::discrete_distribution<int> cellDist(cellRateVec.begin(), cellRateVec.end());
    const int parentCellIndex = cellDist(rng);
    Cell<DIM> &parentCell = cells[parentCellIndex];
    std::discrete_distribution<int> spDist(parentCell.cellBirthRateBySpecies.begin(),
                                           parentCell.cellBirthRateBySpecies.end());
    const int s = spDist(rng);
    const int parentIdx = std::uniform_int_distribution<int>(0, parentCell.population[s] - 1)(rng);
    auto &parentPos = parentCell.coords[s][parentIdx];
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    const double radius = evalBirthKernel(s, u);
    auto dir = randomUnitVector(rng);
    for (int d = 0; d < DIM; d++) {
        dir[d] *= radius;
    }
    std::array<double, DIM> childPos;
    for (int d = 0; d < DIM; d++) {
        childPos[d] = parentPos[d] + dir[d];
    }
    spawn_at(s, childPos);
}

template <int DIM>
void Grid<DIM>::kill_random() {
    if (total_death_rate < 1e-12) {
        return;
    }
    std::vector<double> cellRateVec(total_num_cells);
    for (int i = 0; i < total_num_cells; i++) {
        cellRateVec[i] = cells[i].cellDeathRate;
    }
    std::discrete_distribution<int> cellDist(cellRateVec.begin(), cellRateVec.end());
    const int cellIndex = cellDist(rng);
    Cell<DIM> &cell = cells[cellIndex];
    std::discrete_distribution<int> spDist(cell.cellDeathRateBySpecies.begin(),
                                           cell.cellDeathRateBySpecies.end());
    const int s = spDist(rng);
    if (cell.population[s] == 0) {
        return;
    }
    std::discrete_distribution<int> victimDist(cell.deathRates[s].begin(),
                                               cell.deathRates[s].end());
    const int victimIdx = victimDist(rng);
    const std::array<int, DIM> cIdx = unflattenIdx(cellIndex);
    kill_at(s, cIdx, victimIdx);
}

template <int DIM>
void Grid<DIM>::make_event() {
    const double sumRate = total_birth_rate + total_death_rate;
    if (sumRate < 1e-12) {
        return;
    }
    event_count++;
    std::exponential_distribution<double> expDist(sumRate);
    const double dt = expDist(rng);
    time += dt;
    const double r = std::uniform_real_distribution<double>(0.0, sumRate)(rng);
    const bool isBirth = (r < total_birth_rate);
    if (isBirth) {
        spawn_random();
    } else {
        kill_random();
    }
}

template <int DIM>
void Grid<DIM>::run_events(int events) {
    for (int i = 0; i < events; i++) {
        if (std::chrono::system_clock::now() >
            init_time + std::chrono::duration<double>(realtime_limit)) {
            realtime_limit_reached = true;
            return;
        }
        make_event();
    }
}

template <int DIM>
void Grid<DIM>::run_for(double duration) {
    const double endTime = time + duration;
    while (time < endTime) {
        if (std::chrono::system_clock::now() >
            init_time + std::chrono::duration<double>(realtime_limit)) {
            realtime_limit_reached = true;
            return;
        }
        make_event();
        if (total_birth_rate + total_death_rate < 1e-12) {
            return;
        }
    }
}

template <int DIM>
std::vector<std::vector<std::array<double, DIM>>> Grid<DIM>::get_all_particle_coords() const {
    std::vector<std::vector<std::array<double, DIM>>> result(M);
    for (const auto &cell : cells) {
        for (int s = 0; s < M; ++s) {
            result[s].insert(result[s].end(), cell.coords[s].begin(), cell.coords[s].end());
        }
    }
    return result;
}

template <int DIM>
std::vector<std::vector<double>> Grid<DIM>::get_all_particle_death_rates() const {
    std::vector<std::vector<double>> result(M);
    for (const auto &cell : cells) {
        for (int s = 0; s < M; ++s) {
            result[s].insert(result[s].end(), cell.deathRates[s].begin(), cell.deathRates[s].end());
        }
    }
    return result;
}

template class Grid<1>;
template class Grid<2>;
template class Grid<3>;
