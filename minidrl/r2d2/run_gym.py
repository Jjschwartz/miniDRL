"""Run R2D2 on classic gym environments."""

from minidrl.r2d2.r2d2 import run_r2d2
from minidrl.r2d2.utils import parse_r2d2_args


if __name__ == "__main__":
    run_r2d2(parse_r2d2_args())
