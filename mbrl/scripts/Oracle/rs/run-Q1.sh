
for ENV in _pendulum _acrobot _cartpole _mountain_car
do
for DISCOUNT in 1.0 0.99 0.97 0.95 0.9 0.7 0.5 0.25 0.1 0.0
do
    tsp python run.py \
        env=${ENV} \
        seed=$RANDOM name='Random-Shooting/Oracle/DiscountFactorAblation' \
        discount=${DISCOUNT}
done
done
