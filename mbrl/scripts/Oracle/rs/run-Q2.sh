
for ENV in _pendulum _acrobot _cartpole _mountain_car
do
for POPULATION in 500 1000 2000 5000 10000
do
    tsp python run.py \
        env=${ENV} \
        seed=$RANDOM name='Random-Shooting/Oracle/Population-Ablation' \
        population=${POPULATION}
done
done
