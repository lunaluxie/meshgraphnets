# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Plots a CFD trajectory rollout."""

import pickle

import matplotlib.pyplot as plt
from absl import app, flags
from matplotlib import animation
from matplotlib import tri as mtri

FLAGS = flags.FLAGS
flags.DEFINE_string("rollout_path", None, "Path to rollout pickle file")
flags.DEFINE_bool("use_gt", False, "Whether to animate the ground truth")


def main(unused_argv):
    with open(FLAGS.rollout_path, "rb") as fp:
        rollout_data = pickle.load(fp)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    skip = 10
    num_steps = rollout_data[0]["gt_pos"].shape[0]
    num_frames = len(rollout_data) * num_steps // skip

    # compute bounds
    bounds = []
    for trajectory in rollout_data:
        bb_min = trajectory["gt_pos"].min(axis=(0, 1))
        bb_max = trajectory["gt_pos"].max(axis=(0, 1))
        bounds.append((bb_min, bb_max))

    def animate(num):
        step = (num * skip) % num_steps
        traj = (num * skip) // num_steps
        ax.cla()
        bound = bounds[traj]
        ax.set_xlim([bound[0][0], bound[1][0]])
        ax.set_ylim([bound[0][1], bound[1][1]])
        ax.set_zlim([bound[0][2], bound[1][2]])

        pos_key = "gt_pos" if FLAGS.use_gt else "pred_pos"
        pos = rollout_data[traj][pos_key][step]  # pred_pos
        faces = rollout_data[traj]["faces"][step]
        ax.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True)
        ax.set_title("Trajectory %d Step %d" % (traj, step))
        return (fig,)

    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    writervideo = animation.FFMpegWriter(fps=50 // skip)  # realtime using dt=0.02s
    anim.save(
        ".".join(
            [FLAGS.rollout_path.split(".")[0], "gt" if FLAGS.use_gt else "pred", "mp4"]
        ),
        writer=writervideo,
    )


if __name__ == "__main__":
    app.run(main)
