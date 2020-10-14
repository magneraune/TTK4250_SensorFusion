# %% imports
from typing import List

import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gaussparams import GaussParams
from mixturedata import MixtureParameters
import dynamicmodels
import measurementmodels
import ekf
import imm
import pda
import estimationstatistics as estats


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# %% plot config check and style setup

# %matplotlib inline

# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "inline")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


# %% load data and plot
filename_to_load = "data_joyride.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
Ts = loaded_data["Ts"].squeeze()
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]
# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=6, clear=True)


Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 450
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
plt.show(block=False)

# %% play measurement movie. Remember that you can cross out the window
play_movie = False
play_slice = slice(0, K)
if play_movie:
    if "inline" in matplotlib.get_backend():
        print("the movie might not play with inline plots")
    fig2, ax2 = plt.subplots(num=2, clear=True)
    sh = ax2.scatter(np.nan, np.nan)
    th = ax2.set_title(f"measurements at step 0")
    mins = np.vstack(Z).min(axis=0)
    maxes = np.vstack(Z).max(axis=0)
    ax2.axis([mins[0], maxes[0], mins[1], maxes[1]])
    plotpause = 0.1
    # sets a pause in between time steps if it goes to fast
    for k, Zk in enumerate(Z[play_slice]):
        sh.set_offsets(Zk)
        th.set_text(f"measurements at step {k}")
        fig2.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(plotpause)


# %% IMM CV-CT model

# measurement model
sigma_z = 30 #10
clutter_intensity = 1e-5#1e-2
PD = 0.85 #0.8
gate_size = 3

# dynamic models
sigma_a_CV = 0.5 #0.5
sigma_a_CV_high = 3
sigma_a_CT = 1.2 #0.5
sigma_omega = 0.5*np.pi/180#0.225#0.3


# markov chain
PI11 = 0.95
PI22 = 0.95
PI33 = 0.90

p10 = 0.8  # initvalue for mode probabilities

PI = np.array([[PI11, (1 - PI11)], [(1 - PI22), PI22]])

PI12 = 0.025
PI13 = 0.025

PI21 = 0.025
PI23 = 0.025

PI31 = 0.05
PI32 = 0.05

PI_cv_ct_cvh = np.array([[PI11, PI12, PI13], [PI21, PI22, PI23], [PI31, PI32, PI33]])

print(PI_cv_ct_cvh)

assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

mean_init = np.array([7000, 3500, 0, 0, 0])
cov_init = np.diag([1000, 1000, 30, 30, 0.1]) ** 2    # THIS WILL NOT BE GOOD: [1000, 1000, 30, 30, 0.1]
mode_probabilities_init = np.array([p10, (1 - p10)])

mode_probabilities_init_CV_CT_CVh = np.array([p10, (1 - p10)/2,(1-p10)/2])


mode_states_init = GaussParams(mean_init, cov_init)

init_imm_state = MixtureParameters(mode_probabilities_init, [mode_states_init] * 2)
init_imm_state_CV_CT_CVh = MixtureParameters(mode_probabilities_init_CV_CT_CVh, [mode_states_init] * 3)
init_ekf_state = GaussParams(mean_init, cov_init)


assert np.allclose(
    np.sum(mode_probabilities_init), 1
), "initial mode probabilities must sum to 1"

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV_high, n=5))


ekf_filters, ekf_filters_cvctcvh = [], []
ekf_filter_1 = ekf.EKF(dynamic_models[0], measurement_model)
ekf_filter_2 = ekf.EKF(dynamic_models[1], measurement_model)
ekf_filter_3 = ekf.EKF(dynamic_models[2], measurement_model)

ekf_filters.append(ekf_filter_1)
ekf_filters.append(ekf_filter_2)

ekf_filters_cvctcvh.append(ekf_filter_1)
ekf_filters_cvctcvh.append(ekf_filter_2)
ekf_filters_cvctcvh.append(ekf_filter_3)

imm_filter = imm.IMM(ekf_filters, PI)
imm_filter_cvctcvh = imm.IMM(ekf_filters_cvctcvh, PI_cv_ct_cvh)

tracker_1 = pda.PDA(ekf_filter_1, clutter_intensity, PD, gate_size) # EKF CV
tracker_2 = pda.PDA(ekf_filter_2, clutter_intensity, PD, gate_size) # EKF CT

tracker_3 = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)
tracker_4 = pda.PDA(imm_filter_cvctcvh, clutter_intensity, PD, gate_size)

trackers = [tracker_1, tracker_2, tracker_3, tracker_4]
names = ["EKF-CV-PDA", "EKF-CT-PDA", "IMM-PDA (CV-CT)","IMM-PDA (CV-CT-CVhigh)"]

#init_imm_pda_state = tracker.init_filter_state(init_immstate)

NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

tracker_update_init = [init_ekf_state, init_ekf_state, init_imm_state, init_imm_state_CV_CT_CVh]
tracker_update_list = np.empty((len(trackers), len(Xgt)), dtype=MixtureParameters)
tracker_predict_list = np.empty((len(trackers), len(Xgt)), dtype=MixtureParameters)
tracker_estimate_list = np.empty((len(trackers), len(Xgt)), dtype=MixtureParameters)
# estimate
Ts = np.insert(Ts,0, 0., axis=0)

x_hat = np.empty((len(trackers), len(Xgt), 5)) 
prob_hat = np.empty((len(trackers), len(Xgt), 2))

NEES = np.empty((len(trackers), len(Xgt), 1))
NEESpos = np.empty((len(trackers), len(Xgt), 1))
NEESvel = np.empty((len(trackers), len(Xgt), 1))


for i, (tracker, name) in enumerate(zip(trackers, names)):
    print("Running: ",name)
    for k, (Zk, x_true_k, Tsk) in enumerate(zip(Z, Xgt, Ts)):
        if k == 0:
            tracker_predict = tracker.predict(tracker_update_init[i], Tsk)
        else:
            tracker_predict = tracker.predict(tracker_update, Tsk)
        tracker_update = tracker.update(Zk, tracker_predict)

        # You can look at the prediction estimate as well
        tracker_estimate = tracker.estimate(tracker_update)

        NEES[i][k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(4))
        NEESpos[i][k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2))
        NEESvel[i][k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2, 4))

        tracker_predict_list[i][k]= tracker_predict
        tracker_update_list[i][k] = (tracker_update)
        tracker_estimate_list[i][k] = (tracker_estimate)

    x_hat[i] = np.array([est.mean for est in tracker_estimate_list[i]])
    if i == 2:
        prob_hat_double = np.array([upd.weights for upd in tracker_update_list[i]])
    if i == 3:
        prob_hat_triple = np.array([upd.weights for upd in tracker_update_list[i]])

# calculate a performance metrics
posRMSE = np.empty((len(trackers),1), dtype=float)
velRMSE = np.empty((len(trackers),1), dtype=float)
peak_pos_deviation = np.empty((len(trackers),1), dtype=float)
peak_vel_deviation = np.empty((len(trackers),1), dtype=float)

for i,_ in enumerate(trackers):
    poserr = np.linalg.norm(x_hat[i,:, :2] - Xgt[:, :2], axis=1)
    velerr = np.linalg.norm(x_hat[i,:, 2:4] - Xgt[:, 2:4], axis=1)
    posRMSE[i] = np.sqrt(
        np.mean(poserr ** 2)
    )  # not true RMSE (which is over monte carlo simulations)
    velRMSE[i] = np.sqrt(np.mean(velerr ** 2))
    # not true RMSE (which is over monte carlo simulations)
    peak_pos_deviation[i] = poserr.max()
    peak_vel_deviation[i] = velerr.max()



# consistency
confprob = 0.90
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

confprob = confprob
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K

ANEESpos = np.empty((len(trackers),1))
ANEESvel = np.empty((len(trackers),1))
ANEES = np.empty((len(trackers),1))

for i,_ in enumerate(trackers):
    ANEESpos[i] = np.mean(NEESpos)
    ANEESvel[i] = np.mean(NEESvel)
    ANEES[i] = np.mean(NEES)

# plots

def create_tsk(Ts):
    tsk = np.empty(Ts.shape)
    val = 0.0
    for i, element in enumerate(Ts):
        val += element
        tsk[i] = val
    return tsk
tsk = create_tsk(Ts)

# trajectory for EKF-PDA

fig6, ax6 = plt.subplots(1, 2, num=1, clear=True)
ax6[0].plot(*Xgt.T[:2], '-', label="$ground  truth$")
for i, (_, name) in enumerate(zip(trackers, names)):
    if i < 2:
        ax6[0].plot(*x_hat[i].T[:2], '--', label=name)

title = ''
for i,(name) in enumerate(names):
    if i < 2:
        title += name + "\n"
        title += f"RMSE(pos, vel) = ({posRMSE[i,0]:.3f}, {velRMSE[i,0]:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation[i,0]:.3f}, {peak_vel_deviation[i,0]:.3f}) \n\n"


ax6[0].text(0.95, 0.01, title,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax6[0].transAxes,
        color='black', fontsize=10  )
ax6[0].set_xlim([4500, 7500])

ax6[0].set_title("Trajectory from EKF-PDA trackers", fontsize=15)
ax6[0].axis("equal")
ax6[0].legend(loc='upper left')


# trajectory for IMM-PDA

ax6[1].plot(*Xgt.T[:2], '-', label="$ground  truth$")

axins = zoomed_inset_axes(ax6[1],2,loc=6)
axins.plot(*Xgt.T[:2], '-')
axins.scatter(*Z_plot_data.T, color="C1", alpha=0.5, label="measurements")

axins.set_title("zigzag movement",fontsize=7)
for i, (_, name) in enumerate(zip(trackers, names)):
    if i > 1:
        ax6[1].plot(*x_hat[i].T[:2], '--', label=name)
        axins.plot(*x_hat[i].T[:2], '--')


axins.set_xlim([5700,6050])
axins.set_ylim([1900, 2200])
axins.legend(loc='upper left', fontsize=6)
#mark_inset(ax6[1], axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.setp(axins, xticks=[], yticks=[])

title = ''
for i,(name) in enumerate(names):
    if i > 1:
        title += name + "\n"
        title += f"RMSE(pos, vel) = ({posRMSE[i,0]:.3f}, {velRMSE[i,0]:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation[i,0]:.3f}, {peak_vel_deviation[i,0]:.3f}) \n\n"


ax6[1].text(0.95, 0.01, title,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax6[1].transAxes,
        color='black', fontsize=10  )
ax6[1].set_xlim([4500, 7500])

ax6[1].set_title("Trajectory from IMM-PDA trackers", fontsize=15)
ax6[1].axis("equal")
ax6[1].legend(loc='upper left')


fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
axs3[0].set_title("Mode probabilities for IMM-PDA with two modes")
axs3[1].set_title("Mode probabilities for IMM-PDA with three modes")

# probabilities
two_modes = ['CV','CT']
double = axs3[0].plot(tsk, prob_hat_double)
axs3[0].legend(double, two_modes)

three_modes = ("CV","CT","CV high")
triple = axs3[1].plot(tsk, prob_hat_triple)
axs3[1].legend(triple, three_modes) 


axs3[1].set_ylim([0, 1])
axs3[1].set_ylabel("mode probability")
axs3[1].set_xlabel("time [s]")
axs3[0].set_ylabel("mode probability")
axs3[0].set_xlabel("time [s]")

axs3[1].annotate('strong turn caught by large noise CV model', xy=(453, 0.56), xytext=(160, 0.7), fontsize=12,
            arrowprops=dict(facecolor='black', shrink=0.15))


# NEES

fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
fig4.suptitle('EKF-PDA NEESes with confidence bounds')

axs4[0].plot(tsk, NEESpos[0], label='EKF-PDA (CV)')
axs4[0].plot([0, (K - 1) * Ts.mean()], np.repeat(CI2[None], 2, 0), "--r")
axs4[0].set_ylabel("NEES pos")
inCIpos = np.mean((CI2[0] <= NEESpos[0]) * (NEESpos[0] <= CI2[1]))
title_NEESpos = (f"\n{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI for EKF-PDA (CV)")

axs4[1].plot(tsk, NEESvel[0], label='EKF-PDA (CV)')
axs4[1].plot([0, (K - 1) * Ts.mean()], np.repeat(CI2[None], 2, 0), "--r")
axs4[1].set_ylabel("NEES vel")
inCIvel = np.mean((CI2[0] <= NEESvel[0]) * (NEESvel[0] <= CI2[1]))
title_NEESvel = (f"\n{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI for EKF-PDA (CV)")

axs4[2].plot(tsk, NEES[0], label='EKF-PDA (CV)')
axs4[2].plot([0, (K - 1) * Ts.mean()], np.repeat(CI4[None], 2, 0), "--r")
axs4[2].set_ylabel("NEES")
inCI = np.mean((CI4[0] <= NEES[0]) * (NEES[0] <= CI4[1]))
title_NEES = (f"\n{inCI*100:.1f}% inside {confprob*100:.1f}% CI for EKF-PDA (CV)")
axs4[2].set_xlabel("time [s]")

#print(f"ANEESpos = {ANEESpos[0]:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
#print(f"ANEESvel = {ANEESvel[0]:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
#print(f"ANEES = {ANEES[0]:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

#fig7, axs7 = plt.subplots(3, sharex=True, num=7, clear=True)
axs4[0].plot(tsk, NEESpos[1], label='EKF-PDA (CT)')

inCIpos = np.mean((CI2[0] <= NEESpos[1]) * (NEESpos[1] <= CI2[1]))

title_NEESpos += (f"\n {inCIpos*100:.1f}% inside {confprob*100:.1f}% CI for EKF-PDA (CT)")
axs4[0].set_title(title_NEESpos)

axs4[1].plot(tsk, NEESvel[1], label='EKF-PDA (CT)')
inCIvel = np.mean((CI2[0] <= NEESvel[1]) * (NEESvel[1] <= CI2[1]))
title_NEESvel += (f"\n{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI for EKF-PDA (CT)")
axs4[1].set_title(title_NEESvel)

axs4[2].plot(tsk, NEES[1], label='EKF-PDA (CT)')
inCI = np.mean((CI4[0] <= NEES[1]) * (NEES[1] <= CI4[1]))
title_NEES += (f"\n{inCI*100:.1f}% inside {confprob*100:.1f}% CI for EKF-PDA (CT)")
axs4[2].set_title(title_NEES)

axs4[0].legend(loc="upper left")
axs4[1].legend(loc="upper left")
axs4[2].legend(loc="upper left")


#print(f"ANEESpos = {ANEESpos[1]:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
#print(f"ANEESvel = {ANEESvel[1]:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
#print(f"ANEES = {ANEES[1]:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")




##new

fig7, axs7 = plt.subplots(3, sharex=True, num=8, clear=True)
axs7[0].plot(tsk, NEESpos[2], label='IMM-PDA (CV-CT)')

axs7[0].plot([0, (K - 1) * Ts.mean()], np.repeat(CI2[None], 2, 0), "--r")
axs7[0].set_ylabel("NEES pos")
inCIpos = np.mean((CI2[0] <= NEESpos[2]) * (NEESpos[2] <= CI2[1]))
title_NEESpos2 = (f"\n{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI for IMM-PDA (CV-CT)")

axs7[1].plot(tsk, NEESvel[2], label='IMM-PDA (CV-CT)')
axs7[1].plot([0, (K - 1) * Ts.mean()], np.repeat(CI2[None], 2, 0), "--r")
axs7[1].set_ylabel("NEES vel")
inCIvel = np.mean((CI2[0] <= NEESvel[2]) * (NEESvel[2] <= CI2[1]))
title_NEESvel2 = (f"\n{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI for IMM-PDA (CV-CT)")

axs7[2].plot(tsk, NEES[2], label='IMM-PDA (CV-CT)')
axs7[2].plot([0, (K - 1) * Ts.mean()], np.repeat(CI4[None], 2, 0), "--r")
axs7[2].set_ylabel("NEES")
inCI = np.mean((CI4[0] <= NEES[2]) * (NEES[2] <= CI4[1]))
title_NEES2 = (f"\n{inCI*100:.1f}% inside {confprob*100:.1f}% CI for IMM-PDA (CV-CT)")




fig7.suptitle('IMM-PDA NEESes with confidence bounds')

axs7[0].plot(tsk, NEESpos[3], label='IMM-PDA (CV-CT-CVhigh)')

inCIpos = np.mean((CI2[0] <= NEESpos[3]) * (NEESpos[3] <= CI2[1]))
title_NEESpos2 += (f"\n{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI for IMM-PDA (CV-CT-CVhigh)")
axs7[0].set_title(title_NEESpos2)

axs7[1].plot(tsk, NEESvel[3], label='IMM-PDA (CV-CT-CVhigh)')
inCIvel = np.mean((CI2[0] <= NEESvel[3]) * (NEESvel[3] <= CI2[1]))
title_NEESvel2 += (f"\n{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI for IMM-PDA (CV-CT-CVhigh)")
axs7[1].set_title(title_NEESvel2)

axs7[2].plot(tsk, NEES[3], label='IMM-PDA (CV-CT-CVhigh)')
inCI = np.mean((CI4[0] <= NEES[3]) * (NEES[3] <= CI4[1]))
title_NEES2 += (f"\n{inCI*100:.1f}% inside {confprob*100:.1f}% CI for IMM-PDA (CV-CT-CVhigh)")
axs7[2].set_title(title_NEES2)

axs7[0].legend(loc="upper left")
axs7[1].legend(loc="upper left")
axs7[2].legend(loc="upper left")


axs7[2].set_xlabel("time [s]")


# errors

fig5, axs5 = plt.subplots(2,2, num=5, clear=True)
for i, (_, name) in enumerate(zip(trackers, names)):
    if i < 2:
        axs5[0][0].plot(tsk, np.linalg.norm(x_hat[i,:, :2] - Xgt[:, :2], axis=1), label=name)
        axs5[1][0].plot(tsk, np.linalg.norm(x_hat[i,:, 2:4] - Xgt[:, 2:4], axis=1), label=name)
    if i >= 2:
        axs5[0][1].plot(tsk, np.linalg.norm(x_hat[i,:, :2] - Xgt[:, :2], axis=1), label=name)
        axs5[1][1].plot(tsk, np.linalg.norm(x_hat[i,:, 2:4] - Xgt[:, 2:4], axis=1), label=name)

axs5[0][0].set_title("RMSE for EKF-PDAs")
axs5[0][0].set_ylabel("position error")
axs5[1][0].set_ylabel("velocity error")
axs5[0][0].set_xlabel("time [s]")
axs5[1][0].set_xlabel("time [s]")

axs5[0][1].set_title("RMSE for IMM-PDAs")
axs5[0][1].set_ylabel("position error")
axs5[1][1].set_ylabel("velocity error")
axs5[0][1].set_xlabel("time [s]")
axs5[1][1].set_xlabel("time [s]")

axs5[0][0].legend(loc="upper left")
axs5[0][1].legend(loc="upper left")
axs5[1][0].legend(loc="upper left")
axs5[1][1].legend(loc="upper left")
plt.show()
