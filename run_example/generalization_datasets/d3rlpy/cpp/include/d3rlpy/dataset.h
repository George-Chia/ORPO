#include <vector>
#include <memory>

namespace d3rlpy {

  using namespace std;

  struct CTransition {
    vector<int> observation_shape;
    int action_size;
    unsigned char* observation_i;
    float* observation_f;
    int action_i;
    float* action_f;
    float reward;
    unsigned char* next_observation_i;
    float* next_observation_f;
    float terminal;
    shared_ptr<CTransition> prev_transition;
    shared_ptr<CTransition> next_transition;
  };
  struct CTransitionRecord2rewards {
    vector<int> observation_shape;
    int action_size;
    unsigned char* observation_i;
    float* observation_f;
    int action_i;
    float* action_f;
    float reward;
    float modified_reward;
    unsigned char* next_observation_i;
    float* next_observation_f;
    float terminal;
    shared_ptr<CTransitionRecord2rewards> prev_transition;
    shared_ptr<CTransitionRecord2rewards> next_transition;
  };

}
