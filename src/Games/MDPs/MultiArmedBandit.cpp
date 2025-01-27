#include "../../../include/Games/MDPs/MultiArmedBandit.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>

using namespace std;
using namespace MAB;

Model::Model(const std::vector<std::pair<double,double>>& arm_distributions, int arm_copies){
    for(int i = 0; i < arm_copies; i++)
        this->arm_distributions.insert(this->arm_distributions.end(), arm_distributions.begin(), arm_distributions.end());
    for(int i = 0; i < this->arm_distributions.size(); i++)
        actions.push_back(i);
}

void Model::printState(ABS::Gamestate* state) {
    auto* WFState = dynamic_cast<Gamestate*>(state);
    if (!WFState) return;

    std::cout << "Multi Armed Bandit has no internal state." << std::endl;
}

ABS::Gamestate* Model::getInitialState(std::mt19937& rng){
    auto* state = new MAB::Gamestate();
    return state;
}

int Model::getNumPlayers() {
    return 1;
}

size_t Gamestate::hash() const {
    return 0;
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

std::pair<std::vector<double>,std::pair<int,double>> Model::applyAction_(ABS::Gamestate* uncasted_state, int action, std::mt19937& rng) {
    auto* state = dynamic_cast<MAB::Gamestate*>(uncasted_state);

    std::normal_distribution<double> dist(arm_distributions[action].first, arm_distributions[action].second);
    double reward = dist(rng);
    state->terminal = true;

    return {{reward}, {0, 1}};
}