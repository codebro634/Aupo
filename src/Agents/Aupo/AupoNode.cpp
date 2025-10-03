#include <random>
#include <algorithm>
#include "../../../include/Agents/Aupo/AupoNode.h"

using namespace AUPO;

AupoNode::AupoNode(
    ABS::Model* model,
    ABS::Gamestate* state,
    int depth,
    std::mt19937& rng
) : model(model), state(state), depth(depth)
{
    visits = 0;
    untried_actions = state->terminal? std::vector<int>() : model->getActions(state);
    std::ranges::shuffle(untried_actions.begin(), untried_actions.end(), rng);
}

std::string AupoNode::toString(){
    //print attributes
    std::string str = "Node: " + std::to_string(id) + "\n";
    str += "Depth: " + std::to_string(depth) + "\n";
    str += "Visits: " + std::to_string(visits) + "\n";
    str += "Rewards: ";
    for (auto action : tried_actions)
    {
        str += std::to_string(getActionValues(action)) + "/" + std::to_string(getActionVisits(action)) + " ";
    }
    str += "\n";
    return str;
}

int AupoNode::popUntriedAction(){
    int a = untried_actions.back();
    untried_actions.pop_back();
    tried_actions.push_back(a);
    children[a] = {};
    return a;
}

void AupoNode::addVisits(int visits)
{
    this->visits+= visits;
}

void AupoNode::addActionVisits(const int action, int visits)
{
    action_visits[action] += visits;
}

void AupoNode::addActionValues(const int action, double value)
{
    action_values[action] += value;
}

bool AupoNode::isFullyExpanded() const
{
    return untried_actions.empty();
}

int AupoNode::getDepth() const
{
    return depth;
}

ABS::Model* AupoNode::getModel() const
{
    return model;
}


double AupoNode::getActionValues(const int action, bool init_default){
    return init_default? action_values[action] : action_values.at(action);
}

int AupoNode::getActionVisits(const int action, bool init_default){
    return init_default? action_visits[action] : action_visits.at(action);
}

std::vector<int>* AupoNode::getTriedActions()
{
    return &tried_actions;
}

std::map<int, gsToNodeMap<AupoNode*>>* AupoNode::getChildren()
{
    return &children;
}

int AupoNode::getVisits() const
{
    return visits;
}


bool AupoNode::isTerminal() const
{
    return state->terminal;
}

ABS::Gamestate* AupoNode::getStateCopy() const
{
    return model->copyState(state);
}

int AupoNode::getPlayer() const
{
    return state->turn;
}
