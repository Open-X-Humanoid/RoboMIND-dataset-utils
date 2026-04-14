## RoboMIND-dataset-utils
We developed a set of data quality check scripts. After performing a full scan across benchmark1_0/1_1/1_2, we discovered that the puppet/end_effector data anomalies (where the values remain almost constant) are primarily concentrated in the following three data categories: h5_ur_1rgb, h5_simulation, and h5_sim_franka_3rgb.
The EE (end effector) data for the other models (h5_franka_1rgb/3rgb, h5_agilex_3rgb, h5_tienkung_xsens_1rgb, h5_franka_fr3_dual) is normal.

### Root Cause
It is highly likely that a blocking issue occurred at the collection end during data acquisition, which prevented the end effector pose from updating synchronously with the frames.

### Impact on Training
Our data is primarily oriented towards VLA (Vision-Language-Action) training, where the core feature utilized is the joint_position.If the end effector pose is strictly required for your specific use case, it can be recalculated via forward kinematics using the corresponding model's URDF combined with the joint_position.

This repository includes the following contents:

- Check script

- joint2pose script

- detailed CSV report
