# Greenhouse Navigation: D* Lite with MPC and TD3 Local Control

Global replanning and local control for a mobile robot in a greenhouse with
walking workers. D* Lite plans on an inflated occupancy grid and repairs the plan
incrementally when the LiDAR sees a worker step into the aisle. Two local
controllers follow the plan, a nonlinear MPC and a TD3 policy, and are compared
on the same scenarios against a pure-pursuit baseline and an end-to-end TD3
policy that receives no global plan.

> **About this repository.** This is a re-implementation, written in 2026, of the
> greenhouse navigation part of my 2024-25 undergraduate project (National Natural
> Science Foundation of China). The original ran TD3 in Gazebo on a ROS 1 stack; that
> code is not included here. This version rebuilds the same greenhouse layout in a
> lightweight 2D simulator, adds the classical D* Lite + MPC pipeline and dynamic
> obstacles, and every number in this README is produced by the code in this
> repository.

> **Status: in progress.** Sections marked `TBD` have not been measured yet.

<!-- demo GIF goes here -->

## The greenhouse

The default layout reproduces the original Gazebo world `melon.world`: a 24 m x 19 m
house with four rows of 7 m x 1 m planting racks, a 4 m central aisle and 3 m aisles
between rows. Rack occupancy is 12.3% of the floor. Coordinates were read from the
original world file.

For a 0.6 m robot this layout is not dense. The aisle-width sweep below narrows the
aisles step by step so that "dense, repetitive structure" is a measured condition
rather than a description.

## Methods

| | Global | Local |
|---|---|---|
| D* + pure pursuit | D* Lite | geometric pursuit (baseline) |
| D* + MPC | D* Lite | nonlinear MPC, CasADi / IPOPT |
| D* + TD3 | D* Lite | TD3 policy, LiDAR + next waypoint |
| TD3 end-to-end | none | TD3 policy, LiDAR + final goal |

## Scope

**Does:** unicycle kinematics with velocity and acceleration limits for a Scout
Mini footprint, 2D LiDAR ray casting, workers that walk and pause, D* Lite with
incremental repair, fixed-seed evaluation with confidence intervals.

**Does not:** simulate wheel slip, terrain, or 3D structure; use a physics engine;
run on a real robot. Collision ends an episode rather than being simulated.

## Verification

`tests/test_dstar.py` checks D* Lite against an independent Dijkstra on the same grid:
path cost after the initial search, after cells are blocked on the current path, and
after they are released. A regression test for a static-cell release bug is included,
and was confirmed to fail when the fix is removed.

## Results

`TBD`

## Reproducing

`TBD`
