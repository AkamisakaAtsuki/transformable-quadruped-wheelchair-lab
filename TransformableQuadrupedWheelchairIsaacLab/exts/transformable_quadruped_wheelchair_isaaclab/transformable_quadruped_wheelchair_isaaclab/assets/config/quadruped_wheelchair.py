import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from transformable_quadruped_wheelchair_isaaclab import TQW_PATH

p_gain = 500  # 150
d_gain = 5    # 10

B2_MOTOR_CALF = DCMotorCfg(
    joint_names_expr=[".*_calf_joint"],
    effort_limit=320,
    saturation_effort=320,
    velocity_limit=14,
    friction=0.0,
    stiffness=p_gain,
    damping=d_gain,
)

B2_MOTOR_DEFAULT = DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
    effort_limit=200,
    saturation_effort=200,
    velocity_limit=23,
    friction=0.0,
    stiffness=p_gain,
    damping=d_gain,
)

TQW_TIRE = DelayedPDActuatorCfg(
    joint_names_expr=[".*Tire1_joint"],
    effort_limit=20000,  # もともとは200だった
    velocity_limit=55.0,
    min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
    max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
    stiffness=0.0,
    damping=0.3,
    friction=0.0,
    armature=0.0,
)

Quadruped_Wheelchair_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TQW_PATH}/assets/usd/4-legged_wheelchair.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            "slider_joint": 0.325,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": B2_MOTOR_DEFAULT,
        "calf": B2_MOTOR_CALF,
        "chair": DCMotorCfg(
            joint_names_expr=["ChairArm_joint", "SittingChairAngle_joint", "BackSeat_joint", "BottomSeat_joint", "RightArmSupport_joint", "LeftArmSupport_joint", "FootLegSupport_joint"],
            effort_limit=300,
            saturation_effort=300,
            velocity_limit=30.0,
            stiffness=1000.0,
            damping=0,
            friction=0.0,
        ),
        "slider": DCMotorCfg(
            joint_names_expr=["slider_joint"],
            effort_limit=10000,
            saturation_effort=10000,
            velocity_limit=30.0,
            stiffness=10000.0,
            damping=0,
            friction=0.0,
        ),
        "wheel": TQW_TIRE,
        "steering": DCMotorCfg(
            joint_names_expr=["RFUpper2_joint", "LFUpper2_joint", "LRUpper2_joint", "RRUpper2_joint"],
            effort_limit=300,
            saturation_effort=300,
            velocity_limit=30.0,
            stiffness=1000.0,
            damping=0,
            friction=0.0,
        ),
    },
)