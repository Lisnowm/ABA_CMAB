import numpy as np
import matplotlib.pyplot as plt
from new_VirtualChild import VirtualChild
from new_VirtualTherapist import VirtualTherapist

# --- Configuration Section ---
REINFORCERS = [
    # Stickiness: Cost/Effort penalty. Higher = harder to prepare.
    # Satiation_rate: How fast preference drops (Pref * rate).
    # Recovery_rate: How fast preference recovers over time.
    {'name': 'iPad', 'stickiness': 0.7, 'init_pref': 0.9, 'satiation_rate': 0.7, 'recovery_rate': 0.1},
    {'name': 'Chips', 'stickiness': 0.2, 'init_pref': 0.6, 'satiation_rate': 0.4, 'recovery_rate': 0.3},
    {'name': 'Sticker', 'stickiness': 0.0, 'init_pref': 0.3, 'satiation_rate': 0.9, 'recovery_rate': 0.8}
]

TASKS = [
    {'name': 'Easy', 'difficulty': 0.1, 'mastery': 0.1},
    {'name': 'Medium', 'difficulty': 0.4, 'mastery': 0.1},
    {'name': 'Hard', 'difficulty': 0.8, 'mastery': 0.1}
]

# --- Helper Function: Update Task Mastery ---
def update_task_mastery(current_mastery, focus, difficulty, alpha=0.1, beta=0.05):
    """
    Updates the mastery level of a task based on current mastery and emotional response.
    Formula: Mt+1 = Mt + alpha*(1-Mt) + beta*Et + noise
    """
    noise = np.random.normal(0, 0.01)
    base_change = alpha * (1 - current_mastery) + beta * focus + noise
    learning_rate_modifier = 1.0 - (difficulty * 0.8) 
    change = base_change * learning_rate_modifier
    new_mastery = current_mastery + change
    return np.clip(new_mastery, 0.0, 1.0)


# --- Main Simulation Workflow (Random Strategy) ---
def run_random_simulation(num_trials=50):
    # Initialize the environment (Child)
    child = VirtualChild(REINFORCERS)
    
    # We initialize the Therapist class just to keep the structure consistent,
    # but we will NOT use its decision-making logic (choose_reinforcer) in this function.
    therapist = VirtualTherapist(n_arms=len(REINFORCERS), n_features=7)
    
    history_rewards = []
    history_choices = []
    history_regret = []

    # Logging Header
    print(f"{'Trial':<5} | {'Task':<8} | {'Choice':<8} | {'Pref':<5} | {'Emo':<5} | {'Stickness':<5} | {'Reward':<6} | {'Regret':<6}")
    print("-" * 80)

    for t in range(num_trials):
        # 1. Task Selection (Randomly select a task)
        task_idx = np.random.choice(len(TASKS))
        current_task = TASKS[task_idx]
        
        # 2. Get Child State (Current emotion, fatigue, preferences, etc.)
        child_state = child.get_state()

        # 3. Calculate the Best Possible Reward (Oracle / Theoretical Best)
        # This is used ONLY to calculate Regret, not for decision making.
        potential_rewards = []

        for i in range(len(REINFORCERS)): 
            expected_r = child.get_expected_reward(
                task_difficulty=current_task['difficulty'], 
                chosen_reinforcer_idx=i,
                reinforcer_data=REINFORCERS[i]
            )
            potential_rewards.append(expected_r)

        max_theoretical_reward = max(potential_rewards)
        
        # --- KEY CHANGE FOR RANDOM STRATEGY ---
        # 4. Random Selection
        # Instead of asking the therapist algorithm, we pick a random index.
        chosen_idx = np.random.randint(0, len(REINFORCERS))
        
        chosen_item = REINFORCERS[chosen_idx]
        expected_reward_of_chosen = potential_rewards[chosen_idx] 
        
        # 5. Child Reacts (Generate Emotion & Compliance)
        pref_at_decision = child_state['preferences'][chosen_idx]
        
        discrete_emo, discrete_com, emotion, compliance, focus = child.react(
            current_task['difficulty'], 
            chosen_idx
        )

        # 6. Calculate Actual Reward Received
        W_FOC = 10.0
        W_COMPLIANCE = 8.0
        W_EMOTION = 5.0
        W_STICKINESS = 8.0
        
        actual_reward = (W_FOC * focus) + \
                        (W_COMPLIANCE * discrete_com) + \
                        (W_EMOTION * discrete_emo) - \
                        (W_STICKINESS * REINFORCERS[chosen_idx]['stickiness'])

        # 7. Calculate Regret
        # Regret = Best Possible Outcome - Outcome of our Random Choice
        instant_regret = max_theoretical_reward - expected_reward_of_chosen
        
        history_rewards.append(actual_reward)
        history_choices.append(chosen_idx)
        history_regret.append(instant_regret)
        
        # --- KEY CHANGE FOR RANDOM STRATEGY ---
        # 8. No Algorithm Update
        # Since this is a random strategy, there are no weights to update.
        # We skip: therapist.update_strategy(...)
        
        # 9. Update Context (Post-Task updates)
        # a) Update Task Mastery
        TASKS[task_idx]['mastery'] = update_task_mastery(
            current_task['mastery'], 
            focus,
            current_task['difficulty']
        )
        # b) Update Child Internal States
        child.update_internal_states(chosen_idx)
        
        # Logging
        print(f"{t:<5} | "
        f"{current_task['name']:<8} | "
        f"{chosen_item['name']:<8} | "
        f"{pref_at_decision:.2f}  | "
        f"{emotion:.2f}   | "          
        f"{chosen_item['stickiness']:.1f}   | "
        f"{actual_reward:.2f}   | "
        f"{instant_regret:.2f}")

    return history_choices, history_rewards, history_regret

if __name__ == "__main__":
    # Run the random simulation
    choices, rewards, regrets = run_random_simulation(50)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot Cumulative Reward
    plt.subplot(1, 3, 1)
    plt.plot(np.cumsum(rewards), label='Random Strategy', color='gray')
    plt.title("Cumulative Reward (Random Baseline)")
    plt.xlabel("Trial")
    plt.ylabel("Total Reward")
    plt.legend()

    # Plot Cumulative Regret (Should be linear/increasing for Random)
    plt.subplot(1, 3, 2)
    plt.plot(np.cumsum(regrets), color='red', label='Random Strategy')
    plt.title("Cumulative Regret (Random Baseline)")
    plt.xlabel("Trial")
    plt.legend()
    
    # Plot Choice Distribution (Should be roughly equal/uniform)
    plt.subplot(1, 3, 3)
    plt.hist(choices, bins=np.arange(len(REINFORCERS)+1)-0.5, rwidth=0.8, color='gray')
    plt.xticks(range(len(REINFORCERS)), [r['name'] for r in REINFORCERS])
    plt.title("Choice Distribution")
    
    plt.tight_layout()
    plt.show()