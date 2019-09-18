#!/bin/bash

pkill -f -SIGTERM "param-synth"
pkill -f -SIGTERM "simulate"
pkill -f -SIGTERM "simulators/BioNetGen"
