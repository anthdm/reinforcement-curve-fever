import pygame
from game import Game
from fever import Curve
from pygame.math import Vector2 as Vec2
import random
import numpy as np
import matplotlib.pyplot as plt

game = Game(640, 480) 
curve = Curve(Vec2(100, 100)) 
game.curve = curve

while(1):
    action = random.choice([0, 1, 2])
    frame, reward = game.step(action)
