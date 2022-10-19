---
sidebar_position: 1
---

# Tutorial: Deploy a Taichi program in C++

Taichi is currently embedded in Python frontend and users from its easy-to-write syntax which enables fast prototyping. However Python is still heavy for industrial applications which have strict performance and packaging constraints. Taichi offers a simple workflow so that you can develop in Python frontend and seamless deployment.

This tutorial provides a step-by-step guide to deploy a Taichi program in an C++ application.

## 0. Overview



Ndarray

Cgraph

C-API

a graph:
- what you need
- what it does

Assuming you have a working Taichi program that you want to deploy, there're only two steps in this workflow:
- Select and save your compiled artifacts on your host machine with a python environment
- Load and run Taichi kernels in your application.

Let's zoom in and break these into smaller pieces.

host     application




## 1. Rewrite and make sure it runs using cgraph

## 2. Save kernels on disk


======

how to integrate taichi into your application


Compiled taichi artifacts  --load--> taichi runtime lib  --link--> your application

## 3. Get Taichi Runtime Library

Get

## 4. Run taichi kernels in your application

## 4. Limitations


## FAQ

- import/export device

- supported archs (ios)

- How to debug?

- dev cap / host arch?

- versioning

- render pipeline?

- More examples?

- raw spirv shaders?

- build from source?
