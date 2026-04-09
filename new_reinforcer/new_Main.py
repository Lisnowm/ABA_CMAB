import numpy as np
import matplotlib.pyplot as plt
from new_VirtualChild import VirtualChild
import copy
from new_VirtualTherapist import VirtualTherapist


# --- Configuration Section ---
REINFORCERS = [
    # Stickiness: Penalty term (Cost). Higher means harder to prepare/clean up.
    # Satiation_rate: Smaller value means preference drops faster (e.g., Pref * 0.7).
    # Recovery_rate: Larger value means preference recovers faster over time.
    {'name': 'iPad', 'transition': 0.4, 'init_pref': 0.9, 'satiation_rate': 0.7, 'recovery_rate': 0.1, 'fatigue_recovery':0.02},
    {'name': 'Chips', 'transition': 0.2, 'init_pref': 0.6, 'satiation_rate': 0.4, 'recovery_rate': 0.3, 'fatigue_recovery':0.25},
    {'name': 'Sticker', 'transition': 0.0, 'init_pref': 0.3, 'satiation_rate': 0.9, 'recovery_rate': 0.8, 'fatigue_recovery':0.10}
]
'''
REINFORCERS = [
    # transition: transition reinforcer to task.
    # Satiation_rate: Smaller value means preference drops faster (e.g., Pref * 0.7).
    # Recovery_rate: Larger value means preference recovers faster over time.
    {'name': 'iPad', 'transition': 0.4, 'init_pref': 0.9, 'satiation_rate': 0.7, 'recovery_rate': 0.1, 'fatigue_recovery':0.02},
    {'name': 'Chips', 'transition': 0.2, 'init_pref': 0.1, 'satiation_rate': 0.5, 'recovery_rate': 0.3, 'fatigue_recovery':0.02},
    {'name': 'Sticker', 'transition': 0.0, 'init_pref': 0.1, 'satiation_rate': 0.4, 'recovery_rate': 0.4, 'fatigue_recovery':0.02}
]
'''

TASKS = [
    {'name': 'Easy', 'difficulty': 0.1, 'mastery': 0.1},
    {'name': 'Medium', 'difficulty': 0.4, 'mastery': 0.1},
    {'name': 'Hard', 'difficulty': 0.8, 'mastery': 0.1}
]


# --- Helper Function: Update Task Mastery ---
def update_task_mastery(current_mastery, focus, difficulty,alpha=0.1, beta=0.05):
    """
    Updates the mastery level of a task based on current mastery and focus.
    Formula: Mt+1 = Mt + alpha*(1-Mt) + beta*Ft + noise
    """
    noise = np.random.normal(0, 0.01)
    base_change = alpha * (1 - current_mastery) + beta * focus + noise
    learning_rate_modifier = 1.0 - (difficulty * 0.8) 
    change = base_change * learning_rate_modifier
    new_mastery = current_mastery + change
    return np.clip(new_mastery, 0.0, 1.0)


# --- Main Simulation Workflow ---
def run_simulation(profile_name,strategy_mode = 'rl',num_trials=50):

    # strategy mode = rl, greedy and random

    current_reinforcers = copy.deepcopy(REINFORCERS)
    current_tasks = copy.deepcopy(TASKS)

    print(f"\n--- Running Simulation for Profile: {profile_name.upper()} ---")
    child = VirtualChild(profile_name, current_reinforcers)
    #child = VirtualChild(profile_name, REINFORCERS)
    # Features: [Bias, Emotion, Fatigue, Mastery, Preference] -> 5 dimensions
    # therapist = VirtualTherapist(n_arms=len(REINFORCERS), n_features=7)
    #therapist = VirtualTherapist(n_arms=len(current_reinforcers), n_features=7)
    therapist = VirtualTherapist(n_arms=len(current_reinforcers), n_features=9)
    
    # algorithm related
    history_rewards = []
    history_choices = []
    history_regret = []

    # task related
    history_emotion=[]
    history_focus = []
    history_compliance = []
    history_fatigue = []
    history_resistance = []
    history_mastery = []
    
    print(f"{'Trial':<5} | {'Task':<10} | {'Choice':<10} | {'Pref':<6} | {'Emo':<6} | {'dEmo':<5} | {'dCom':<5} | {'Resist':<6} | {'Reward':<6} | {'Regret':<6}")
    print("-" * 95)

    for t in range(num_trials):
        # 1. Task Selection (Randomly select a task)
        #task_idx = np.random.choice(len(TASKS))
        #current_task = TASKS[task_idx]

        task_idx = np.random.choice(len(current_tasks))
        current_task = current_tasks[task_idx]
        
        # 2. Get Child State (Current emotion, fatigue, preferences, etc.)
        child_state = child.get_state()

        # 3. Calculate the Best Possible Reward (Oracle / Theoretical Best) / for calculation of regret
        potential_rewards = []

        for i in range(len(current_reinforcers)): 
            # Call the helper method directly. 
            # No need to copy-paste math formulas here!
            expected_r = child.get_expected_reward(
                task_difficulty=current_task['difficulty'], # Assuming you have current_task
                chosen_reinforcer_idx=i,
                reinforcer_data=current_reinforcers[i]
            )
            potential_rewards.append(expected_r)

        # The best reward we COULD have gotten (Optimal Arm)
        max_theoretical_reward = max(potential_rewards)
        #optimal_arm_idx = np.argmax(potential_rewards)  the index of the best reinforcer

        # 4 Strategy Selection
        context_vector = None # Only needed for RL update
    
        if strategy_mode == 'random':
            # Random
            chosen_idx = np.random.randint(0, len(current_reinforcers))
            
        elif strategy_mode == 'greedy':
            # Choose the reinforcer with highest preference
                perceived_prefs = [p + np.random.normal(0, 0.1) for p in child_state['preferences']]
                chosen_idx = np.argmax(perceived_prefs)
            
        else:
            # strategy_mode == 'rl':
            # reinforced learning model
            
            suggested_idx, context_vector = therapist.choose_reinforcer(
                child_state, 
                current_task['mastery'],
                current_task['difficulty'],
                current_reinforcers
            )

            if t < 3:
                chosen_idx = np.argmax(child_state['preferences'])
            else:
                chosen_idx = suggested_idx

        chosen_item = current_reinforcers[chosen_idx]
        expected_reward_of_chosen = potential_rewards[chosen_idx]

        '''
        
        # 4. Therapist Chooses Reinforcer - only for RL algorithm
        # Pass in the mastery level of the current task as context
        chosen_idx, context_vector = therapist.choose_reinforcer(
            child_state, 
            current_task['mastery'],
            current_task['difficulty'],
            current_reinforcers
        )
        chosen_item = current_reinforcers[chosen_idx]
        expected_reward_of_chosen = potential_rewards[chosen_idx] 
        '''
        
        # 5. Child Reacts (Generate Emotion & Compliance)
        # Note: We capture the preference *before* the reaction affects it
        pref_at_decision = child_state['preferences'][chosen_idx]
        
        # Discrete version
        discrete_emo, discrete_com, discrete_res_score, emotion, compliance, resistance, focus, fatigue = child.react(
            current_task['difficulty'], 
            chosen_idx
        )

        # Continuous version
        #emotion,compliance,focus = child.react(
        #    current_task['difficulty'], 
        #    chosen_idx
        #)

        # after reinforcement update, continuous version
        history_emotion.append(emotion)
        history_compliance.append(compliance)
        history_focus.append(focus)
        history_fatigue.append(fatigue)
        

        # 6. Calculate Actual Reward Received
        # Must use the SAME weights as get_expected_reward
        W_FOC = 10.0
        W_COMPLIANCE = 8.0
        W_EMOTION = 5.0
        W_resistance = 8.0
        #W_transition = 3.0

        # actual_reward = (W_FOC * focus) + \
        #                (W_COMPLIANCE * compliance) + \
        #                (W_EMOTION * emotion) - \
        #                (W_transition * REINFORCERS[chosen_idx]['transition'])
        
        actual_reward = (W_FOC * focus) + \
                        (W_COMPLIANCE * discrete_com) + \
                        (W_EMOTION * discrete_emo) - \
                        (W_resistance * discrete_res_score)



        # 7. Calculate Regret
        # Regret = Best Possible Expected Reward - Actual Reward Received
        # Note: Sometimes Actual Reward > Max Theoretical Reward due to positive noise.
        # In strict theory, Regret = Max Expected - Expected of Chosen.
        # But in many simulations, Regret = Max Expected - Actual is used for tracking.
        # Let's use the standard definition: Regret >= 0
        instant_regret = max_theoretical_reward - expected_reward_of_chosen

        # Optional: Keep cumulative regret
        # total_regret += instant_regret
        
        history_rewards.append(actual_reward)
        history_choices.append(chosen_idx)
        history_regret.append(instant_regret)
        history_resistance.append(resistance)

        
        # 8. Update Algorithm (Train the therapist)
        if strategy_mode == 'rl':
            therapist.update_strategy(context_vector, actual_reward)
        
        # 9. Update Context (Post-Task updates)
        # a) Update Task Mastery based on performance/emotion
        current_tasks[task_idx]['mastery'] = update_task_mastery(
            current_task['mastery'], 
            focus,
            current_task['difficulty']
        )

        # b) Update Child Internal States (Satiation decreases pref, Fatigue increases)
        recovery_val = chosen_item['fatigue_recovery']
        child.update_internal_states(chosen_idx, recovery_val)

        avg_mastery = np.mean([t['mastery'] for t in current_tasks])
        history_mastery.append(avg_mastery)
        
        # Logging
        # print(f"{t:<5} | {current_task['name']:<8} | {chosen_item['name']:<8} | "
        #      f"{pref_at_decision:.2f}  | {emotion:.2f}  | {chosen_item['transition']:.1f}   | {actual_reward:.2f}   | {instant_regret:.2f}")

        # Logging
        print(f"{t:<5} | "
        f"{current_task['name']:<8} | "
        f"{chosen_item['name']:<8} | "
        f"{pref_at_decision:.2f}  | "
        f"{emotion:.2f}   | "      
        f"{discrete_emo:<5.1f} | "        
        f"{discrete_com:<5.1f} | "    
         f"{resistance:<6.2f} |   | "
        f"{actual_reward:.2f}   | "
        f"{instant_regret:.2f}")

    # return history_choices, history_rewards, history_regret
    return {
        'rewards': history_rewards,    
        'compliance': history_compliance,
        'emotion': history_emotion,
        'focus': history_focus,
        'regrets': history_regret,
        'fatigue': history_fatigue,
        'choices': history_choices,
        'mastery': history_mastery  

    }
'''
if __name__ == "__main__":
    # 【修改】只运行 'normal' 用户
    profiles = ['normal'] 
    strategies = ['random', 'greedy', 'rl']
    colors = {'random': 'gray', 'greedy': 'red', 'rl': 'blue'}
    styles = {'random': ':', 'greedy': '--', 'rl': '-'}

    all_results = {}

    print("Running Simulations for Single User (Normal)...")
    for p in profiles:
        all_results[p] = {}
        for s in strategies:
            all_results[p][s] = run_simulation(p, strategy_mode=s, num_trials=50)

    p = 'normal'
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Performance Analysis: Profile '{p.upper()}' (Standard Parameters)", fontsize=16)
    
    # 1. Cumulative Reward
    ax = axes[0, 0]
    for s in strategies:
        data = np.cumsum(all_results[p][s]['rewards'])
        ax.plot(data, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=2)
    ax.set_title("Cumulative Reward (Higher is Better)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative Regret
    ax = axes[0, 1]
    for s in strategies:
        data = np.cumsum(all_results[p][s]['regrets'])
        ax.plot(data, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=2)
    ax.set_title("Cumulative Regret (Lower is Better)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Regret Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Compliance (Smoothed)
    ax = axes[1, 0]
    for s in strategies:
        raw_data = all_results[p][s]['compliance']
        smoothed = np.convolve(raw_data, np.ones(5)/5, mode='valid')
        ax.plot(smoothed, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=1.5)
    ax.set_title("Compliance Trend (Smoothed)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Compliance (-1 to 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Emotion Trend
    ax = axes[1, 1]
    for s in strategies:
        raw_data = all_results[p][s]['emotion']
        smoothed = np.convolve(raw_data, np.ones(5)/5, mode='valid') 
        ax.plot(smoothed, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=1.5)
    ax.set_title("Emotion Trend (Smoothed)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Emotion (-1 to 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


'''

# --- 5. Execution & Visualization ---
if __name__ == "__main__":
    profiles = ['novelty_seeker', 'low_endurance', 'rigid']
    strategies = ['random', 'greedy', 'rl']

    # picture parameters:
    colors = {'random': 'gray', 'greedy': 'red', 'rl': 'blue'}
    styles = {'random': ':', 'greedy': '--', 'rl': '-'}

    # Store results
    all_results = {}

print("Running Simulations...")
for p in profiles:
        all_results[p] = {}
        for s in strategies:
            print(f"  Profile: {p}, Strategy: {s}")
            # Run simulation
            all_results[p][s] = run_simulation(p, strategy_mode=s, num_trials=500)

import matplotlib.pyplot as plt
import numpy as np

profiles = list(all_results.keys()) 
n_profiles = len(profiles)



# fig, axes = plt.subplots(n_profiles, 4, figsize=(24, 5 * n_profiles))
fig, axes = plt.subplots(n_profiles, 5, figsize=(24, 5 * n_profiles))

fig.suptitle("Comprehensive Analysis: All Profiles vs. All Metrics", fontsize=20, y=0.98)


for i, p in enumerate(profiles):
    
    # --- 1. Cumulative Reward  ---
    ax = axes[i, 0]
    for s in strategies:
        data = np.cumsum(all_results[p][s]['rewards'])
        ax.plot(data, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=2)
    

    if i == 0: ax.set_title("Cumulative Reward (Higher is Better)", fontsize=14, fontweight='bold')
    
    ax.set_ylabel(f"Profile: {p.upper()}\nTotal Reward", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 2. Cumulative Regret  ---
    ax = axes[i, 1]
    for s in strategies:
        data = np.cumsum(all_results[p][s]['regrets'])
        ax.plot(data, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=2)
    
    if i == 0: ax.set_title("Cumulative Regret (Lower is Better)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Regret Loss")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 3. Compliance (Smoothed)  ---
    ax = axes[i, 2]
    for s in strategies:
        raw_data = all_results[p][s]['compliance']
        smoothed = np.convolve(raw_data, np.ones(10)/10, mode='valid')
        ax.plot(smoothed, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=1.5)
    
    if i == 0: ax.set_title("Compliance Trend (Smoothed)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Compliance Rate")
    ax.set_ylim(-0.1, 1.1) 
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 4. Emotion Trend (Smoothed)  ---
    ax = axes[i, 3]
    for s in strategies:
        raw_data = all_results[p][s]['emotion']
        smoothed = np.convolve(raw_data, np.ones(10)/10, mode='valid')
        ax.plot(smoothed, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=1.5)
    
    if i == 0: ax.set_title("Emotion Trend (Smoothed)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Emotion Score")
    ax.set_ylim(-0.1, 1.1) 
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 5. Task mastery ---
    ax = axes[i, 4]
    for s in strategies:
        data = all_results[p][s]['mastery']
        ax.plot(data, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=2)
    if i == 0: ax.set_title("Avg Task Mastery", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mastery Level (0-1)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nGenerating Action Distribution Plots...")
    
fig2, axes2 = plt.subplots(n_profiles, 1, figsize=(15, 4 * n_profiles), sharex=False)
if n_profiles == 1: axes2 = [axes2]
    
fig2.suptitle("Strategy Visualization: Action Selection Over Time (Greedy vs. RL)", fontsize=18, y=0.96)
    
reinforcer_names = [r['name'] for r in REINFORCERS]
y_ticks = range(len(reinforcer_names))
    
for i, p in enumerate(profiles):
    ax = axes2[i]
        

    greedy_choices = all_results[p]['greedy']['choices']
    rl_choices = all_results[p]['rl']['choices']
        
    ax.scatter(range(len(greedy_choices)), [y - 0.15 for y in greedy_choices], 
                   color='red', marker='|', s=80, alpha=0.5, label='Greedy Strategy')
        
    ax.scatter(range(len(rl_choices)), [y + 0.15 for y in rl_choices], 
                   color='blue', marker='|', s=80, alpha=0.5, label='RL Strategy')
        

    ax.set_title(f"Profile: {p.upper()} - Action Switching Pattern", fontsize=14, fontweight='bold')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(reinforcer_names, fontsize=12)
    ax.set_ylim(-0.5, len(reinforcer_names) - 0.5)
    ax.set_xlabel("Trial Number (Time)", fontsize=10)
    ax.set_xlim(0, 500) 
        

    ax.grid(True, axis='x', alpha=0.3) 
    ax.grid(True, axis='y', linestyle='--', alpha=0.5) 
    ax.legend(loc='upper right', frameon=True)
        
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

'''
for p in profiles:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Performance Analysis: Profile '{p.upper()}'", fontsize=16)
        
        # 1. Cumulative Reward
        ax = axes[0, 0]
        for s in strategies:
            # [中文注释] 绘制累积奖励曲线
            data = np.cumsum(all_results[p][s]['rewards'])
            ax.plot(data, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=2)
        ax.set_title("Cumulative Reward (Higher is Better)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Cumulative Regret
        ax = axes[0, 1]
        for s in strategies:
            # [中文注释] 绘制累积遗憾曲线（越低越好）
            data = np.cumsum(all_results[p][s]['regrets'])
            ax.plot(data, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=2)
        ax.set_title("Cumulative Regret (Lower is Better)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Regret Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Compliance (Smoothed)
        ax = axes[1, 0]
        for s in strategies:
            raw_data = all_results[p][s]['compliance']
            # [中文注释] 使用滑动平均（Moving Average）平滑曲线，使趋势更清晰
            smoothed = np.convolve(raw_data, np.ones(5)/5, mode='valid')
            ax.plot(smoothed, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=1.5)
        ax.set_title("Compliance Trend (Smoothed)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Compliance (-1 to 1)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Emotion Trend
        ax = axes[1, 1]
        for s in strategies:
            raw_data = all_results[p][s]['emotion']
            # [中文注释] 同样对情绪数据进行平滑处理
            smoothed = np.convolve(raw_data, np.ones(5)/5, mode='valid') 
            ax.plot(smoothed, label=f"{s.upper()}", color=colors[s], linestyle=styles[s], linewidth=1.5)
        ax.set_title("Emotion Trend (Smoothed)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Emotion (-1 to 1)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
'''

'''
    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # 1. Cumulative Reward (Performance)
    ax = axes[0, 0]
    for p in profiles:
        ax.plot(np.cumsum(results[p]['rewards']), label=p)

    ax.set_title("Cumulative Reward (Therapy Success)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative Regret (Algorithm Performance)
    ax = axes[0, 1]
    for p in profiles:
        ax.plot(np.cumsum(results[p]['regrets']), label=p, linestyle='--')
    ax.set_title("Cumulative Regret (Lower is Better)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Regret Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Fatigue Progression
    ax = axes[1, 0]
    for p in profiles:
        ax.plot(results[p]['fatigue'], label=p)
    ax.set_title("Fatigue Accumulation")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Fatigue Level (0-1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Compliance Levels
    ax = axes[1, 1]
    for p in profiles:
        # Smooth the line for better visualization
        smoothed = np.convolve(results[p]['compliance'], np.ones(5)/5, mode='valid')
        ax.plot(smoothed, label=p)
    ax.set_title("Compliance Trend (Smoothed)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Compliance (-1 to 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Choice Distribution (Bar Chart)
    ax = axes[2, 0]
    bar_width = 0.25
    indices = np.arange(3) # 3 items: iPad, Chips, Praise
    
    for i, p in enumerate(profiles):
        counts = [results[p]['choices'].count(0), 
                  results[p]['choices'].count(1), 
                  results[p]['choices'].count(2)]
        ax.bar(indices + i*bar_width, counts, bar_width, label=p)
        
    ax.set_title("Reinforcer Selection Distribution")
    ax.set_xticks(indices + bar_width)
    ax.set_xticklabels(['iPad (High Val)', 'Chips (Med)', 'Sticker (Low)'])
    ax.legend()

    # Emotion 
    ax = axes[2, 1]
    for p in profiles:
        ax.plot(results[p]['emotion'], label=p)
    ax.set_title("Emotional Accumulation")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Emotional Level (-1 - 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

'''
'''
if __name__ == "__main__":
    choices, rewards, regrets = run_simulation(50)
    
    # Simple Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot Cumulative Reward
    plt.subplot(1, 3, 1)
    plt.plot(np.cumsum(rewards))
    plt.title("Cumulative Reward")
    plt.xlabel("Trial")
    plt.ylabel("Total Reward")

    # Plot Cumulative Regret (Lower is better)
    plt.subplot(1, 3, 2)
    plt.plot(np.cumsum(regrets), color='red')
    plt.title("Cumulative Regret")
    plt.xlabel("Trial")
    
    # Plot Choice Distribution
    plt.subplot(1, 3, 3)
    plt.hist(choices, bins=[0,1,2,3], align='left', rwidth=0.8)
    plt.xticks([0,1,2], ['iPad', 'Chips', 'Sticker'])
    plt.title("Choice Distribution")
    
    plt.tight_layout()
    plt.show()
'''