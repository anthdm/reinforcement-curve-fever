import pygame
import cv2
from fever import Curve

background = (0,0,0)
frame_size = 80

class Game():
    def __init__(self, w, h):
        self.screen = pygame.display.set_mode((w, h))
        self.curve = Curve(pygame.math.Vector2(50, 50)) 
        self.screen_width = w
        self.screen_height = h
        self.best_score = 0
        self.latest_score = 0

    def step(self, action):
        reward = self.update(action)
        frame = self.draw()
        return frame, reward

    def update(self, action):
        pygame.event.pump()
        self.handle_off_screen()

        reward = 0.0
        if self.curve.check_collision():
            self.reset_game()
            reward = -1

        self.curve.update(action)
        return reward

    def draw(self):
        self.screen.fill(background)
        self.draw_curve()

        # Copy raw pixels from the surface into a 3D array.
        current_frame = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()

        return self.pre_process_frame(current_frame)

    def draw_curve(self):
        for pos in self.curve.history:
            pygame.draw.circle(self.screen, self.curve.color, (int(pos.x), int(pos.y)), self.curve.radius, 0)

        if not self.curve.append_history:
            pygame.draw.circle(self.screen, self.curve.color, (int(self.curve.position.x), int(self.curve.position.y)), self.curve.radius, 0)

    def handle_off_screen(self):
        if self.curve.position.x > self.screen_width: 
            self.curve.position.x = 0
        if self.curve.position.x < 0:
            self.curve.position.x = self.screen_width
        if self.curve.position.y < 0:
            self.curve.position.y = self.screen_height
        if self.curve.position.y > self.screen_height:
            self.curve.position.y = 0

    def reset_game(self):
        self.curve.position = pygame.math.Vector2(50, 50)
        score = len(self.curve.history)
        print("Score : {} | best was {}").format(score, self.best_score)
        if score > self.best_score:
            self.best_score = score
        self.latest_score = score
        self.curve.history = []

    def pre_process_frame(self, raw_frame):
        frame = cv2.cvtColor(cv2.resize(raw_frame, (frame_size, frame_size)), cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        return frame

