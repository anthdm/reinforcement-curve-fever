import pygame
import math
import random
from pygame.math import Vector2 as Vec2

curve_d = 5
speed = 1.5
max_turn_rate = 8.
color = (239,79,153)

cut_percentage = 30
cut_len = 30

def circles_collided(c1, c2):
    dx = c1.x - c2.x
    dy = c1.y - c2.y

    dist = math.sqrt((dx * dx) + (dy * dy))

    if dist < (curve_d * 2):
        return True

    return False

class Curve():
    def __init__(self, pos):
        self.position = pos
        self.radius = curve_d
        self.history = []
        self.angle = 0.0
        self.color = color
        self.timestep = 0
        self.t_cut = 0
        self.append_history = True

    def update(self, action):
        if action[0] == 1:
            self.angle += max_turn_rate
        if action[1] == 1:
            self.angle -= max_turn_rate

        if self.timestep % 300 == 0:
            n = random.randint(0, 100)
            if n <= cut_percentage:
                self.append_history = False

        if self.t_cut == cut_len:
            self.append_history = True 
            self.t_cut = 0

        position = self.position + Vec2(1, 0).rotate(self.angle) * speed
        if self.append_history:
            self.history.append(position)
        else:
            self.t_cut += 1

        self.position = position
        self.timestep += 1

    def check_collision(self):
        if len(self.history) <= 9:
            return False
        for pos in self.history[:len(self.history) - 9]:
            if circles_collided(self.position, pos):
                return True

        return False
