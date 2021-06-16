


for ENV in _pendulum _acrobot _cartpole _mountain_car
do
    tsp python run.py env=${ENV} seed=$RANDOM name='Random-Shooting/Oracle/Bench'
done
