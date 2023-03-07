from deep_q_learning import *
from policy_grad import *

if __name__ == '__main__':
    # Check Run
    pi_deep, qsa, learning_curve = train_deep_q(verbose=True)
    final_profit = interact_test(pi_deep, series_name='test', verbose=True)
    # plt.plot(learning_curve)
    # plt.xlabel("Episode")
    # plt.ylabel("Profit($)")
    # plt.savefig('learning_curve.pdf')
    # plt.show()

    # Train Deep Q-Learning for 20 times
    profits = list()
    curves = list()
    for i in range(20):
        pi_deep, qsa, learning_curve = train_deep_q(verbose=False)
        profits.append(interact_test(pi_deep, series_name='test', verbose=False))
        curves.append(learning_curve)
        print(i, 'Final Profit', profits[-1])
    print('Avg profit', np.mean(profits))
    print('STD profit', np.std(profits))

    # plt.plot(np.mean(curves, axis=0))
    # plt.xlabel("Episode")
    # plt.ylabel("Profit($)")
    # plt.savefig('learning_curve.pdf')
    # plt.show()

    # pi_policy_grad, rpg = train_policy_gradient(verbose=True)
    # final_profit_policy_grad = interact_test(pi_policy_grad, series_name='test', verbose=True)

    profits = list()
    for i in range(20):
        pi_pg, rpg = train_policy_gradient(verbose=False)
        profits.append(interact_test(pi_pg, series_name='test', verbose=False))
        print(i, 'Final Profit', profits[-1])

    print('Avg profit', np.mean(profits))
    print('STD profit', np.std(profits))

    # pi_deep_sarsa, qsa_sarsa = train_deep_sarsa(verbose=True, sarsa=True)
    # final_profit_sarsa = interact_test(pi_deep_sarsa, series_name='test', verbose=True)

    profits_sarsa = list()
    for i in range(20):
        pi_deep_sarsa, qsa_sarsa = train_deep_sarsa(verbose=False, sarsa=True)
        profits_sarsa.append(interact_test(pi_deep_sarsa, series_name='test', verbose=False))
        print(i, 'Final Profit', profits_sarsa[-1])
    print('Avg profit', np.mean(profits_sarsa))
    print('STD profit', np.std(profits_sarsa))

