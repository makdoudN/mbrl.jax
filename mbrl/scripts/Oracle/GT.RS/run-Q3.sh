
for ENV in _pendulum _acrobot _cartpole _mountain_car
do
for HORIZON in 2 5 10 15 20 50
do
    tsp python run.py \
        env=${ENV} \
        seed=$RANDOM name='Random-Shooting/Oracle/Horizon-Ablation' \
        horizon=${HORIZON}
done
done
