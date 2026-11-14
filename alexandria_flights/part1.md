## Part 1: Features That Survive Motion
### Q1.1: Harris Corners
![q1_1_harris.png](results/part1/q1_1_harris.png)
### Q1.2: Descriptors and Matching
1. Cornich Frames:
    - Raw Patch:
![q1_2_corniche_20_21_raw.png](results/part1/q1_2_corniche_20_21_raw.png)

    - Oriented Patch:
![q1_2_corniche_20_21_oriented.png](results/part1/q1_2_corniche_20_21_oriented.png)

2. Loop Frames:
    - Raw Patch:
    ![q1_2_loop_3_132_raw.png](results/part1/q1_2_loop_3_132_raw.png)
    - Oriented Patch:
    ![q1_2_loop_3_132_oriented.png](results/part1/q1_2_loop_3_132_oriented.png)

It can be noticed that when using consecutive images (with negligible change in orientation), the performance is almost similar with a slight degradation in the oriented version: 2 wrong matches out of 375 in the raw patch (error ≈ 0.5%) and 5 wrong matches out of 342 in the oriented patch (error ≈ 1.46%).

However, when using images with significant change in orientation, feature matching fails when applied to the raw patch but shows significant improvement in performance when applied to the oriented patch: 3 wrong matches out of 3 (error = 100%) in the raw patch and 3 wrong matches out of 377 in the oriented patch (error ≈ 0.79%)