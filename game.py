import random
from collections import namedtuple
from enum import Enum

import numpy as np
import pygame


pygame.init()
font = pygame.font.Font('/run/media/biprarshi/COMMON/files/AI/RL-Project/Snake-RL/arial.ttf', 25)

class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

Point = namedtuple("Point", ["x", "y"])

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

def _default_window_size():
    info = pygame.display.Info()
    w = int(info.current_w * 0.8) // BLOCK_SIZE * BLOCK_SIZE
    h = int(info.current_h * 0.8) // BLOCK_SIZE * BLOCK_SIZE
    return w, h

class SnakeGame:
    def __init__(self , w = None , h = None):
        if w is None or h is None:
            w, h = _default_window_size()
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((w,h) , pygame.RESIZABLE)
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head , Point(self.head.x - BLOCK_SIZE , self.head.y) , Point(self.head.x + BLOCK_SIZE , self.head.y)] # points spanned by snake body
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()

    def _resize(self, new_width, new_height):
        self.w = max(new_width // BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE * 10)
        self.h = max(new_height // BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE * 10)
        self.display = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
        if self.food is not None and (self.food.x >= self.w or self.food.y >= self.h):
            self._place_food()

    def _place_food(self):
        x = random.randint(0 , (self.w - BLOCK_SIZE) // BLOCK_SIZE ) * BLOCK_SIZE
        y = random.randint(0 , (self.h - BLOCK_SIZE) // BLOCK_SIZE ) * BLOCK_SIZE

        food = Point(x,y)
        self.food = food

        if self.food in self.snake:
            self.food = None
            self._place_food()

    def play_step(self , action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.VIDEORESIZE:
                self._resize(event.w, event.h)

        self.head , self.direction = self.move(self.head , action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward -= 10
            return reward , game_over , self.score
        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
        else:
            self.snake.pop() # remove tail

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def get_state(self , action):
        feature_vec = {}

        st , st_dir = self.move(self.head , (1, 0 , 0))
        feature_vec["danger_straight"] = self._is_collision(st)
        lf , lf_dir = self.move(self.head , (0, 1 , 0))
        feature_vec["danger_left"] = self._is_collision(lf)
        rt , rt_dir = self.move(self.head , (0, 0 , 1))
        feature_vec["danger_right"] = self._is_collision(rt)

        feature_vec["dir_up"] = self.direction == Direction.UP
        feature_vec["dir_down"] = self.direction == Direction.DOWN
        feature_vec["dir_left"] = self.direction == Direction.LEFT
        feature_vec["dir_right"] = self.direction == Direction.RIGHT

        feature_vec["food_up"] = self.food.y < self.head.y
        feature_vec["food_down"] = self.food.y > self.head.y
        feature_vec["food_left"] = self.food.x < self.head.x
        feature_vec["food_right"] = self.food.x > self.head.x

        return feature_vec


    def state_to_index(self, feature_vec):
        # state vector to bin
        index = 0
        for bit in feature_vec.values():
            index = (index << 1) | int(bit)
        return index   # 0 to 2047


    def move(self , point :Point , action):
        # action - [straight , left , right]
        direction_wheel = [Direction.RIGHT , Direction.DOWN , Direction.LEFT , Direction.UP]
        idx = direction_wheel.index(self.direction)

        if np.array_equal(idx , [1 , 0 , 0]):
            # no change
            new_dir = direction_wheel[idx]
        elif np.array_equal(idx , [0 , 1 , 0]):
            next_idx = (idx + 1) % 4
            new_dir = direction_wheel[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = direction_wheel[next_idx]

        x = point.x
        y = point.y

        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.UP:
            y += BLOCK_SIZE
        else:
            y -= BLOCK_SIZE

        return Point(x,y) , new_dir


    def _is_collision(self , point = None) -> bool:
        if point is None:
            point = self.head
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            # boundary hit
            return True
        if point in self.snake[1:]:
            # hit itself
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
