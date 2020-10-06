"""
This tutorial is based on the excellent medium article of Paul Shen

titled: Differentiable Programming for Accelerated Reinforcement Learning and Optimal Control via Continuous Time Neural ODEs
link: https://medium.com/swlh/neural-ode-for-reinforcement-learning-and-nonlinear-optimal-control-cartpole-problem-revisited-5408018b8d71
"""

import math
import torch
import matplotlib.animation as animation

from torchdyn.models import NeuralDE
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from torchdiffeq import odeint

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def pi_mod(theta):
    return (theta) % (2*math.pi)


class CartPole(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.y = 0.      # y position of cart
        self.m = 1.      # pole mass kg
        self.M = 5.      # cart mass kg
        self.L = 3.      # pole length
        self.g = -9.8    # acceleration constant m/s^2
        self.delta = 5.  # friction of cart

    def state_to_coords(self, state):
        """
        Convert the cart state to point coordinates of both the pendulum and the cart
        """
        x, _, theta, _ = state
        cart_coords = (x, self.y)
        pole_coords = ([x, x + 2*self.L*math.sin(theta)],
                       [self.y, self.y + 2*self.L*math.cos(theta)])
        return cart_coords, pole_coords

    def forward(self, state, u):
        """
        Cart derivative

        params:
            state: state vector of format (x,dx,theta,dtheta)
            u: force
        """
        # x, dx, theta, dtheta = state.clone()
        _, dx, theta, dtheta = (state[:, 0], state[:, 1],
                                state[:, 2], state[:, 3])
        du = torch.zeros_like(state)

        # these equations are written assuming that theta=0 is down and theta=pi is up
        theta = pi_mod(theta - math.pi)

        du[:, 0] = dx
        du[:, 1] = (
            (-self.m**2*self.L**2*self.g*torch.cos(theta)*torch.sin(theta)
             + self.m*self.L**2
             * (self.m*self.L*dtheta**2*torch.sin(theta) - self.delta*dx)
             + self.m*self.L**2*u)
            / (self.m*self.L**2*(self.M+self.m*(1-torch.cos(theta)**2)))
        )

        du[:, 2] = dtheta
        du[:, 3] = (
            ((self.m+self.M)*self.m*self.g*self.L*torch.sin(theta)
             - self.m*self.L*torch.cos(theta)
             * (self.m*self.L*dtheta**2*torch.sin(theta) - self.delta*dx)
             + self.m*self.L*torch.cos(theta)*u)
            / (self.m*self.L**2*(self.M+self.m*(1-torch.cos(theta)**2)))
        )

        return du

    def sample_state(self, n_samples):
        """
        Returns n_samples of states the cartpole can be in

        """
        x = (torch.rand(n_samples)*-1)*1.
        dx = (torch.rand(n_samples)*-1)*1.
        theta = torch.rand(n_samples)*2*math.pi
        dtheta = (torch.rand(n_samples)*-1)*1.
        return torch.stack([x, dx, theta, dtheta]).T

    def numerically_integrate(
            self, state, u=0., T=1, time_steps=200, method='dopri5'):
        """
        Numerical integration of the dynamical system, used as a baseline

        """

        t = torch.linspace(0, T, time_steps).to(device)
        state = state.to(device)
        return odeint(
            lambda _, x: self.forward(x, u), state, t, method=method
        ).cpu()[: , 0, :]

class SmartAgent(torch.nn.Module):
    """
    Controller, it controls the cartpole deciding what action to take
    in every state u_t.

    """
    def __init__(self, cartpole):
        super().__init__()
        self.cartpole = cartpole

        # "brain" of the controller
        self.f = torch.nn.Sequential(
                torch.nn.Linear(3, 16),
                torch.nn.ELU(),
                torch.nn.Linear(16, 16),
                torch.nn.ELU(),
                torch.nn.Linear(16, 1)
        ).to(device)

    def forward(self, state):
        """
        Derivative of the controller

        """

        _, _, theta, dtheta = (
            state[:, 0], state[:, 1], state[:, 2], state[:, 3])

        # predict a change in force
        # we only use relevant information to simplify the problem
        controller_input = torch.stack([
            torch.cos(theta),
            torch.sin(theta),
            dtheta
        ]).T.to(device)
        force = self.f(controller_input)[:, 0]

        # observe change in system
        du = self.cartpole(state, force)

        return du

class IdleAgent(torch.nn.Module):
    def __init__(self, cartpole):
        super().__init__()
        self.cartpole = cartpole

    def forward(self, state):
        """
        Derivative of the controller

        """

        # this agent is not in the mood today, it is not predicting any force
        force = torch.tensor(0., device=device)

        # observe change in system
        du = self.cartpole(state, force)

        return du


class NeuralOde(torch.nn.Module):
    """
    A wrapper of the continuous neural network that represents the ODE.

    """
    def __init__(self, cartpole, controller, method='dopri5'):
        super().__init__()
        self.cartpole, self.controller = cartpole, controller

        self.model_of_dyn_system = NeuralDE(
            controller, sensitivity='adjoint', solver=method
        ).to(device)

    def final_state_loss(self, state):
        _, dx, theta = state[:, 0], state[:, 1], state[:, 2]

        # get theta in [-pi,+pi]
        theta = pi_mod(theta + math.pi) - math.pi

        return 4*theta**2 + torch.abs(dx)

    def train(self, n_epochs=100, batch_size=200, lr_patience=10,
              early_stop_patience=20, epsilon=0.1):
        optimizer = torch.optim.Adam(
            self.model_of_dyn_system.parameters(), lr=.1)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=lr_patience, factor=0.5)

        steps_since_plat, last_plat = 0, 0
        for i in range(n_epochs):
            optimizer.zero_grad()

            # setup training scenario
            start_state = cartpole.sample_state(batch_size).to(device)

            # run simulation
            final_state = self.model_of_dyn_system(start_state)

            # evaluate performance
            loss = self.final_state_loss(final_state)
            step_loss = loss.mean()
            print("epoch: {}, loss: {}: ".format(i, step_loss))

            loss.sum().backward()
            optimizer.step()
            scheduler.step(step_loss)

            # if stuck on minimum, stop
            delta_loss = abs(last_plat - step_loss.data)
            if ((steps_since_plat >= early_stop_patience) and
                (delta_loss <= epsilon)):
                break
            elif abs(last_plat - step_loss.data) > epsilon:
                last_plat, steps_since_plat = step_loss, 0
            steps_since_plat += 1

    def trajectory(self, state, T=1, time_steps=200):
        """
        Data trajectory from t = 0 to t = T

        """

        state = state.to(device)
        t = torch.linspace(0, T, time_steps).to(device)

        # integrate and remove batch dim
        traj = self.model_of_dyn_system.trajectory(state, t)
        return traj.detach().cpu()[:, 0, :]


def simulate_trajectory(cartpole, traj, pause_frames=100, file_name=None):
    """
    plot a simulation

    """

    plt.style.use('seaborn-pastel')

    traj = torch.cat([traj[0].repeat(pause_frames, 1), traj])

    fig = plt.figure()
    ax = plt.axes(xlim=(-20, 20), ylim=(-20, 20))

    # transfer the state to coordinates
    cart_coord, pole_coord = cartpole.state_to_coords(traj[0])

    # plot cart
    cart_xcoord, cart_ycoord = cart_coord
    cart, = ax.plot(
        cart_xcoord-5., cart_ycoord, 's', color='black', markersize=20)
    cart2, = ax.plot(
        cart_xcoord+5., cart_ycoord, 's', color='black', markersize=20)

    # plot pendulum
    pole_xcoord, pole_ycoord = pole_coord
    pole, = ax.plot(*pole_coord, color='saddlebrown', linewidth=3)
    pole2, = ax.plot(pole_xcoord[0], pole_ycoord[0], 'o', color='grey',
                     markersize=5)
    pole3, = ax.plot(2*pole_xcoord[1], 2*pole_ycoord[1], 'o', color='darkgrey', markersize=15)

    # plot state information
    text = ax.text(0, 10,
                   'dx: {:.1f}, theta: {:.1f}' .format(traj[0][1], traj[0][2]),
                   fontsize=12, horizontalalignment='center')
    state_text = ax.text(0, -10, 'Paused', color='red', fontsize=12,
                         horizontalalignment='center')

    # plot rail
    ax.plot((-100, 100), (0, 0), linewidth=1, color='lightgrey')

    def init():
        return update_cartpole_state(0)

    def update_cartpole_state(i):
        (cart_xcoord, cart_ycoord), (pole_coord) = cartpole.state_to_coords(
            traj[i])
        pole_xcoord, pole_ycoord = pole_coord
        cart.set_data(cart_xcoord-1., cart_ycoord)
        cart2.set_data(cart_xcoord+1., cart_ycoord)
        pole.set_data(*pole_coord)
        pole2.set_data(pole_xcoord[0], pole_ycoord[0])
        pole3.set_data(pole_xcoord[1], pole_ycoord[1])
        text.set_text('dx: {:.1f}, theta: {:.1f}'
                      .format(traj[i][1], traj[i][2]))
        if i > pause_frames:
            state_text.set_text('Go!')
            state_text.set_color('Green')
        else:
            state_text.set_text('Paused')
        return cart, cart2, pole, pole2, pole3, text, state_text

    def animate(i):
        return update_cartpole_state(i)

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(traj),
                         interval=20, blit=True)
    if file_name is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='nodes tutorial'),
                        bitrate=1800)
        anim.save('{}.mp4'.format(file_name), writer=writer)
    plt.show()


cartpole = CartPole()
# fixed_traj = cartpole.numerically_integrate(cartpole.sample_state(), T=1, time_steps=10)
# fixed_traj = cartpole.numerically_integrate(torch.tensor([[0.,0.,0.,0.5]]), T=100, time_steps=1000)
# fixed_traj = cartpole.numerically_integrate(torch.tensor([[0.,0.,math.pi,0.]]), u=5., T=100, time_steps=1000)
# simulate_trajectory(cartpole, fixed_traj)

# simulate no action
controller = IdleAgent(cartpole)
NOde = NeuralOde(cartpole, controller)
control_traj = NOde.trajectory(torch.tensor([[0., 0., 0., 0.5]]), T=100,
                               time_steps=1000)
simulate_trajectory(cartpole, control_traj)


# simulate upwards swing
# controller = SmartAgent(cartpole)
controller = torch.load('smart_agent.pt')
NOde = NeuralOde(cartpole, controller)
# NOde.train(1000)

start_state = torch.tensor([[0., 0., math.pi, 0.]]).to(device)
control_traj = NOde.trajectory(start_state, T=3, time_steps=100)
simulate_trajectory(cartpole, control_traj)

# simulate action starting from a random state
for i in range(10):
    start_state = cartpole.sample_state(1)
    control_traj = NOde.trajectory(start_state, T=3, time_steps=100)
    simulate_trajectory(cartpole, control_traj)
