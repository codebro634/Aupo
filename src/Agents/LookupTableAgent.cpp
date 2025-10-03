#include "../../include/Agents/LookupTableAgent.h"
#include <cstdlib>
#include <set>

using namespace LookupAgent;

int LookupTableAgent::getAction(ABS::Model* model, ABS::Gamestate* uncasted_gamestate, std::mt19937& rng) {
    auto gamestate = dynamic_cast<FINITEH::Gamestate*>(uncasted_gamestate);
    if (!gamestate)
        throw std::runtime_error("LookupTableAgent requires a finite Horizon gamestate");
    std::vector<int> actions = model->getActions(gamestate);
    double best_value = std::numeric_limits<double>::lowest();
    int best_action = -42;
    [[maybe_unused]] bool found = false;
    auto dist = std::uniform_real_distribution<double>(0.0, 1.0);
    for(int action : actions) {
        if(Q_map->contains({gamestate,action})) {
            double value = Q_map->at({gamestate,action});
            value += dist(rng) * 1e-6;
            if(value > best_value) {
                best_value = value;
                best_action = action;
                found = true;
            }
        }
    }
    assert (found);
    //best_action = actions[std::uniform_int_distribution<int>(0,actions.size()-1)(rng)];

    return best_action;
}

LookupAgent::LookupTableAgent::LookupTableAgent(ABS::Model* model, std::string filename) {
    assert (dynamic_cast<FINITEH::Model*>(model));
    Q_map = new std::unordered_map<std::pair<FINITEH::Gamestate*,int>, double, VALUE_IT::QMapHash, VALUE_IT::QMapCompare>();
    VALUE_IT::loadQTable(dynamic_cast<FINITEH::Model*>(model), Q_map, filename);
    loaded_from_file = true;
}

LookupAgent::LookupTableAgent::LookupTableAgent(std::unordered_map<std::pair<FINITEH::Gamestate*,int>, double, VALUE_IT::QMapHash, VALUE_IT::QMapCompare>* Q_map) {
   this->Q_map = Q_map;
}

LookupAgent::LookupTableAgent::~LookupTableAgent() {
    if(loaded_from_file) {
        for(auto& [key, value] : *Q_map)
            delete key.first;
        delete Q_map;
    }
}