#include <bits/atomic_base.h>

#include "../../../include/Games/MDPs/SysAdmin.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>

using namespace std;
using namespace SA;

Model::Model(bool deterministic){
    connections = DEFAULT_CONNECTIONS;
    actions = {};
    for(int i = 0; i < connections.size() + 1; i++)
        actions.push_back(i);
    this->deterministic = deterministic;
}

Model::Model(const std::string& fileName, bool deterministic)
{
    std::ifstream file(fileName);  // Open the file

    if (!file) {
        std::cerr << "Unable to open file: " << fileName << std::endl;
        return;
    }

    std::string line;
    int biggest_idx = 0;
    while (std::getline(file, line)) {  // Read each line
        std::vector<int> neighbors;
        std::istringstream iss(line);
        int node;
        while (iss >> node) {  // Extract each integer (node) from the line
            neighbors.push_back(node);
            biggest_idx = std::max(biggest_idx, node);
        }
        connections.push_back(neighbors);  // Add the node's neighbors to the adjacency list
    }

    assert (connections.size() == biggest_idx + 1);  // Check that the number of nodes is correct


    //print connections
     // for (int i = 0; i < connections.size(); ++i) {
     //     std::cout << i << "-> ";
     //     for (int j = 0; j < connections[i].size(); ++j) {
     //         std::cout << connections[i][j] << " ";
     //     }
     //     std::cout << std::endl;
     // }

    for(int i = 0; i < connections.size() + 1; i++)
        actions.push_back(i);

    assert (connections.size() <= 32); // we use a 32-bit int to store the state, hence we can't have more than 32 machines

    this->deterministic = deterministic;
}

double Model::getDistance(const ABS::Gamestate* a, const ABS::Gamestate* b) const {
    const Gamestate* state_a = (Gamestate*) a;
    const Gamestate* state_b = (Gamestate*) b;
    return __builtin_popcount( state_a->machine_statuses ^ state_b->machine_statuses);
}

bool Gamestate::operator==(const ABS::Gamestate& other) const
{
    auto other_state = dynamic_cast<const Gamestate*>(&other);
    return machine_statuses == other_state->machine_statuses;
}

size_t Gamestate::hash() const
{
    return machine_statuses;
}

inline void setIthBit(int& n, int i, bool val){
    if(val)
        n |= (1 << i);
    else
        n &= ~(1 << i);
}

inline bool getIthBit(int n, int i){
    return (n & (1 << i)) != 0;
}

void Model::printState(ABS::Gamestate* state) {
    auto* SAState = dynamic_cast<SA::Gamestate*>(state);
    if (!SAState) return;

    for (int i = 0; i < connections.size(); ++i) {
        std::cout << "Machine " << i << ": " << (getIthBit(SAState->machine_statuses,i) ? "ON" : "OFF") << std::endl;
    }
}

ABS::Gamestate* Model::getInitialState(int num){
    auto* state = new SA::Gamestate();
    state->machine_statuses = num;
    return state;
}

ABS::Gamestate* Model::getInitialState(std::mt19937& rng){
    return getInitialState((1 << connections.size()) - 1);
}

int Model::getNumPlayers() {
    return 1;
}


ABS::Gamestate* Model::copyState(ABS::Gamestate* uncasted_state) {
    auto state = dynamic_cast<Gamestate*>(uncasted_state);
    auto new_state = new Gamestate();
    *new_state = *state; //default copy constructor should work
    return new_state;
}

std::vector<int> Model::getActions_(ABS::Gamestate* uncasted_state)  {
    return actions;
}

std::pair<int,int> num_neighbors(int i,  std::vector<std::vector<int>>& connections, int& statuses) {
    int num = connections[i].size();
    int num_on = 0;

    for (int j = 0; j < connections[i].size(); ++j) {
        if (getIthBit(statuses,connections[i][j]))
            num_on++;
    }
    return {num, num_on};
}



std::pair<std::vector<double>,std::pair<int,double>> Model::applyAction_(ABS::Gamestate* uncasted_state, int action, std::mt19937& rng) {
            auto* state = dynamic_cast<SA::Gamestate*>(uncasted_state);
            auto old_statuses = state->machine_statuses;

            double p = 1;
            int pow = 1;
            int successor = 0;
            double reward = (action != connections.size())? -REBOOT_COST : 0;
            std::uniform_real_distribution<double> dist(0, 1);

            for(int i = 0; i < connections.size(); i++){
                if(action == i){
                    // n-th action is no-reboot action{
                    setIthBit(state->machine_statuses,i,true);
                }
                else if(getIthBit(state->machine_statuses,i)){
                    auto [num, num_on] = num_neighbors(i, connections, old_statuses);
                    float stay_active_prob = (deterministic? 0.35 : 0.45) + 0.5 * (1.0 + (float) num_on) / (1.0 + (float) num);
                    auto det_rng = std::mt19937(static_cast<unsigned int>(state->num_moves + i));
                    if(dist(deterministic? det_rng : rng) < stay_active_prob)
                        p *= stay_active_prob;
                    else{
                        p *= 1 - stay_active_prob;
                        setIthBit(state->machine_statuses,i,false);
                        successor += pow;
                    }
                    pow *= 2;
                }else if(!deterministic){
                    //reboot with REBOOT_PROB
                    if(dist(rng) < REBOOT_PROB){
                        setIthBit(state->machine_statuses,i,true);
                        p *= REBOOT_PROB;
                        successor += pow;
                    }
                    else
                        p *= 1 - REBOOT_PROB;

                    pow*=2;
                }
                reward += getIthBit(old_statuses,i)? 1 : 0;
            }

            return {{reward}, {successor, p}};
        }