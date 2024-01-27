import json
import os
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from DCIF_fit_leastsq import DCIF
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid', palette='tab10')


class main:
    def __init__(self):
        self.directory = input('Input Directory Path: ')

        dqs, filenames, stErr = self.get_params()

        self.filenames = np.array(sorted(filenames))
        self.stErr = np.array(sorted(stErr))
        self.dqs = np.array(dqs)

        self.qloc = 0

    def get_params(self):
        # Function reads a directory, and returns the names of the datafile and standard error file. ALso returns the
        #  dq value for each file.

        filenames = []
        stErr = []
        dqs = []

        for root, dirs, files in os.walk(self.directory, topdown=True):
            if len(files) != 3:
                continue

            filename = root + '/' + sorted(files)[-1]
            error_name = root + '/' + sorted(files)[0]
            json_name = root + '/' + sorted(files)[1]
            print(json_name)

            with open(json_name, 'r') as j:
                json_data = json.loads(j.read())
                dqs.append(json_data["MetadataForThisDICF"]["dq"])

            filenames.append(filename)
            stErr.append(error_name)

        return dqs, filenames, stErr

    def run(self):
        # Method to be called in order to run the program.

        method = input('Do you want to perform operations on a single file or a directory? (S/D) ')

        if method == 's' or method == 'S':
            self.single_file_operations()

        elif method == 'd' or method == 'D':
            self.multi_file_operations()

        else:
            print('Invalid Input')

    def single_file_operations(self):
        # When called, prompts the user to input which of the single file operations they want to call
        self.get_q()

        print('These are the single file operations, please select the required one by pressing the corresponding '
              'number: \n 1: Display the raw data with error bars. \n 2: Display the fitted curve \n 3: Find and '
              'display the average velocity \n 4: Find and display the average Diffusion Coefficient.'
              ' \n 5: Find and display the average value of A(q)\n 6: Find and display the average value of the '
              'stretching exponent \n 7: Find and display the average value of tau0 in the stretched exponential ISF'
              ' \n 8: Plot the fit of every q value for a given file')

        while True:
            try:
                operation = int(input('Input operation number: '))
                break

            except ValueError:
                print('Invalid Input. Please try again.')

        if operation == 1:
            self.show_data()

        elif operation == 2:
            self.show_fit()

        elif operation == 3:
            self.show_average_velocity()

        elif operation == 4:
            self.show_average_diff()

        elif operation == 5:
            self.show_average_A()

        elif operation == 6:
            self.show_average_beta()

        elif operation == 7:
            self.show_average_tau0()

        elif operation == 8:
            self.plot_all_fits()

        else:
            print('Invalid Operation')

    @staticmethod
    def get_initial_conditions_swim():
        # Function to allow the user to choose the initial parameters for curve fitting
        default = input('Use Default initial conditions? (y/n) ')
        p0 = np.ones(6)

        if default == 'Y' or default == 'y':
            p0 = np.array([1, 0, 0.3, 10, .3, 5])

        elif default == 'N' or default == 'n':
            p0[0] = input('A: ')
            p0[1] = input('B: ')
            p0[2] = input('D: ')
            p0[3] = input('v: ')
            p0[4] = input('alpha: ')
            p0[5] = input('Z: ')

        else:
            print('Invalid Input')

        return p0

    @staticmethod
    def get_initial_conditions_stretch():
        # Function to allow the user to choose the initial parameters for curve fitting
        default = input('Use Default initial conditions? (y/n) ')
        p0 = np.ones(4)

        if default == 'Y' or default == 'y':
            p0 = np.array([1, 1, 1, 1])

        elif default == 'N' or default == 'n':
            p0[0] = input('A: ')
            p0[1] = input('B: ')
            p0[2] = input('Tau0: ')
            p0[3] = input('beta: ')

        else:
            print('Invalid Input')

        return p0

    def get_q(self):
        # Function to allow user to choose a specific q value
        while True:
            try:
                self.qloc = int(input('Input q value: ')) + 1
                break

            except ValueError:
                print('Invalid input, q should be an integer.')

    def show_data(self):
        # Function to call the dcif show data function
        dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=True)

        dcif.show_data()

    def show_fit(self):
        # Function to show the fit of the Diffusion or the swim diffusion algorithm
        dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=True)

        fit = self.fitting_algorithm_choice()

        if fit == 1:
            dcif.fit_diff()

        elif fit == 2:
            p0 = self.get_initial_conditions_swim()
            dcif.fit_swim(p0)

        elif fit == 3:
            p0 = self.get_initial_conditions_stretch()
            dcif.fit_stretched_exp(p0)

        else:
            print('Invalid input')

    def show_average_velocity(self):
        # Function to show a graph of the velocities over a range of q, and the average.
        dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=True)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()
        p0 = self.get_initial_conditions_swim()

        dcif.plot_Vs(max_q, min_q, False, final_show=True, p0=p0)

    def show_average_diff(self):
        # Function to show a graph of the diffusion coefficients over a range of q, and the average.
        dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=True)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()
        p0 = self.get_initial_conditions_swim()

        fit = self.fitting_algorithm_choice()

        dcif.plot_Ds(max_q, min_q, False, fit, final_choice=True, p0=p0)

    def show_average_beta(self):
        # Function to show a graph of the stretch exponential coefficients over a range of q, and the average.
        dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=True)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()
        p0 = self.get_initial_conditions_stretch()

        dcif.plot_beta(max_q, min_q, False, final_choice=True, p0=p0)

    def show_average_A(self):
        # Function to show a graph of the A(q) values over a range of q, and the average.
        dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=True)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()

        fit = self.fitting_algorithm_choice()

        if fit == 2:
            p0 = self.get_initial_conditions_swim()

        elif fit == 3:
            p0 = self.get_initial_conditions_stretch()

        else:
            p0 = np.array([1, 1, 1])

        dcif.plot_As(max_q, min_q, False, fit, final_choice=True, p0=p0)

    def show_average_tau0(self):
        # Function to show a graph of the stretch exponential tau0 over a range of q, and the average.
        dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=True)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()
        p0 = self.get_initial_conditions_stretch()

        dcif.plot_tau0(max_q, min_q, False, final_choice=True, p0=p0)

    def plot_all_fits(self):
        # Function to show the DCIF of all q values for a given file
        dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=True)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()

        step = self.get_step()

        fit = self.fitting_algorithm_choice()

        fit_times = []
        fit_data = []
        dcif_data = []
        labels = []

        if fit == 1:
            for q in range(min_q, max_q, step):
                print(q)
                dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=q, show=False)
                dcif.fit_diff()

                fit_data.append(dcif.fit_data)
                fit_times.append(dcif.fit_times)
                dcif_data.append(dcif.get_gs(q))
                labels.append(q)

            fit_times = np.array(fit_times)
            fit_data = np.array(fit_data)
            dq = dcif.dq

            self.plot_all_q(fit_data, fit_times, dcif_data, labels, dq)

        elif fit == 2:
            p0 = self.get_initial_conditions_swim()

            for q in range(min_q, max_q, step):
                print(q)
                dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=q, show=False)
                dcif.fit_swim(p0)

                fit_data.append(dcif.fit_data)
                fit_times.append(dcif.fit_times)
                dcif_data.append(dcif.get_gs(q))

                labels.append(q)

            fit_times = np.array(fit_times)
            fit_data = np.array(fit_data)
            dq = dcif.dq

            self.plot_all_q(fit_data, fit_times, dcif_data, labels, dq)

        elif fit == 3:
            p0 = self.get_initial_conditions_stretch()
            for q in range(min_q, max_q, step):
                print(q)
                dcif = DCIF(filename=self.filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=q, show=False)
                dcif.fit_stretched_exp(p0)

                fit_data.append(dcif.fit_data)
                fit_times.append(dcif.fit_times)
                dcif_data.append(dcif.get_gs(q))
                labels.append(q)

            fit_times = np.array(fit_times)
            fit_data = np.array(fit_data)
            dq = dcif.dq

            self.plot_all_q(fit_data, fit_times, dcif_data, labels, dq)
        else:
            print('Invalid input!')

    def plot_all_q(self, fit_data, fit_times, dcif_data, labels, dq):
        # Function to show a graph of DCIFs for multiple q values
        fig, ax = plt.subplots()

        labels = np.array(labels, dtype=int) * dq
        labels = np.round(labels, 6)
        color = iter(cm.rainbow(np.linspace(0, 1, len(fit_data))))

        for i in range(len(fit_data)):
            c = next(color)
            ax.scatter(fit_times[i], dcif_data[i], facecolors='none', color='black')
            # ax.plot(fit_times[i], fit_data[i], label=labels[i], c=c)
            ax.plot(fit_times[i], fit_data[i], label=labels[i], c=c)

        ax.set_xscale('log')
        ax.legend(ncol=2, loc='lower right', title='q($\mu m ^{-1})$')
        ax.set_yscale('log')
        plt.xlabel('Delay Time')
        plt.ylabel('g(q, t)')
        plt.title('DCIFs for \n' + str(self.filenames[0]))
        plt.show()

    @staticmethod
    def get_step():
        # Function to allow user to choose the step when looping through all q values
        while True:
            try:
                step = int(input('Input q step: '))
                break

            except ValueError:
                print('Invalid input, maximum q should be an integer.')

        return step

    @staticmethod
    def max_q_choice(dcif):
        # Helper function to take in the user desired range of q values.
        max_choice = input('Use up to maximum q points? (y/n) ')
        max_q = 0

        if max_choice == 'y' or max_choice == 'Y':
            max_q = dcif.get_q_max()

        elif max_choice == 'n' or max_choice == 'N':
            while True:
                try:
                    max_q = int(input('Input maximum q: '))
                    break

                except ValueError:
                    print('Invalid input, maximum q should be an integer.')
        else:
            print('Invalid input')

        return max_q

    @staticmethod
    def min_q_choice():
        # Helper function to take in the user desired minimum q value.
        while True:
            try:
                min_q = int(input('Input minimum q: '))
                break

            except ValueError:
                print('Invalid input, maximum q should be an integer.')

        return min_q

    def multi_file_operations(self):
        # When called, prompts the user to input which of the multiple file operations they want to call
        self.get_q()

        print('These are the multiple file operations, please select the required one by pressing the corresponding '
              'number: \n 1: Show the fit of each file in the directory. \n 2: Plot the average Velocity for a range '
              'of files \n 3: Plot the average Diffusion Coefficient for a range of files \n 4: Plot the average value '
              'of A(q) for a range of files \n 5: Plot the average tau0 for a range of files \n 6: Plot the average '
              'Stretching Exponent for a range of files')

        while True:
            try:
                operation = int(input('Input operation number: '))
                break

            except ValueError:
                print('Invalid Input. Please try again.')

        if operation == 1:
            self.overplot_fits()

        elif operation == 2:
            self.directory_average_velocity()

        elif operation == 3:
            self.directory_average_diff()

        elif operation == 4:
            self.directory_average_a()

        elif operation == 5:
            self.directory_average_tau0()

        elif operation == 6:
            self.directory_average_beta()

        else:
            print('Invalid Operation')

    def overplot_fits(self):
        # Function to plot the fits of each file in a directory on a single axis.
        filenames = self.filenames
        fit_times = []
        fit_data = []
        names = []

        fit = self.fitting_algorithm_choice()

        if fit == 1:
            self.diffusion_fits(filenames, fit_data, fit_times, names)

        elif fit == 2:
            self.swim_fits(filenames, fit_data, fit_times, names)

        elif fit == 3:
            self.stretch_fits(filenames, fit_data, fit_times, names)

        else:
            print('Invalid Input')

    def swim_fits(self, filenames, fit_data, fit_times, names):
        # Helper function for overplot_fits; over-plots with the SwimDiff algorithm
        p0 = self.get_initial_conditions_swim()
        for i in range(len(filenames)):
            print(filenames[i])

            dcif = DCIF(filename=filenames[i], error_name=self.stErr[i], dq=self.dqs[i], qloc=self.qloc, show=False)

            names.append(dcif.get_name())
            dcif.fit_swim(p0)

            fit_data.append(dcif.fit_data)
            fit_times.append(dcif.fit_times)

        fit_times = np.array(fit_times)
        fit_data = np.array(fit_data)
        names = np.array(names)

        fig, ax = plt.subplots()

        for i in range(len(fit_data)):
            ax.plot(fit_times[i], fit_data[i], label=names[i])

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.xlabel('Time')
        plt.ylabel('g(q, t)')
        plt.title('DCIFs at q = ' + str(self.qloc - 1))
        plt.show()

    def diffusion_fits(self, filenames, fit_data, fit_times, names):
        # Helper function for overplot_fits; over-plots with the Diffusion algorithm
        for i in range(len(filenames)):
            print(filenames[i])
            dcif = DCIF(filename=filenames[i], error_name=self.stErr[i], dq=self.dqs[i], qloc=self.qloc, show=False)

            names.append(dcif.get_name())
            dcif.fit_diff()

            fit_data.append(dcif.fit_data)
            fit_times.append(dcif.fit_times)

        fit_times = np.array(fit_times)
        fit_data = np.array(fit_data)
        names = np.array(names)

        fig, ax = plt.subplots()

        for i in range(len(fit_data)):
            ax.plot(fit_times[i], fit_data[i], label=names[i])

        ax.set_xscale('log')
        ax.legend()
        plt.xlabel('Time')
        plt.ylabel('g(q, t)')
        plt.title('DCIFs at q = ' + str(self.qloc - 1))
        plt.show()

    def stretch_fits(self, filenames, fit_data, fit_times, names):
        # Helper function for overplot_fits; over-plots with the stretched-exp algorithm
        p0 = self.get_initial_conditions_stretch()

        for i in range(len(filenames)):
            print(filenames[i])
            dcif = DCIF(filename=filenames[i], error_name=self.stErr[i], dq=self.dqs[i], qloc=self.qloc, show=False)

            names.append(dcif.get_name())
            dcif.fit_stretched_exp(p0)

            fit_data.append(dcif.fit_data)
            fit_times.append(dcif.fit_times)

        fit_times = np.array(fit_times)
        fit_data = np.array(fit_data)
        names = np.array(names)

        fig, ax = plt.subplots()

        for i in range(len(fit_data)):
            # ax.plot(fit_times[i], fit_data[i], label=names[i])
            ax.plot(np.flip(fit_times)[i], np.flip(fit_data)[i], label=np.flip(names)[i])

        ax.set_xscale('log')
        ax.legend()
        plt.xlabel('Time')
        plt.ylabel('g(q, t)')
        plt.title('DCIFs at q = ' + str(self.qloc - 1))
        plt.show()

    @staticmethod
    def fitting_algorithm_choice():
        # Helper function to take in user choice of fitting algorithm
        print('Input fitting algorithm: \n 1: Diffusion \n 2: SwimDiffusion \n 3: Stretched Exponential')
        while True:
            try:
                fit = int(input())
                break

            except ValueError:
                print('Invalid Input. Please try again.')

        return fit

    def directory_average_a(self):
        # Function to plot the average value for A(q) for files in a directory
        filenames = self.filenames
        names = []

        f = open('const_A.csv', 'w')
        f.close()

        dcif = DCIF(filename=filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=False)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()
        p0 = self.get_initial_conditions_swim()

        for i in range(len(filenames)):
            print(filenames[i])
            dcif = DCIF(filename=filenames[i], error_name=self.stErr[i], dq=self.dqs[i], qloc=self.qloc, show=False)
            names.append(dcif.get_name())
            dcif.plot_As(qmax=max_q, qmin=min_q, show_choice=False, fit=2, final_choice=False, p0=p0)

        with open('const_A.csv', 'r') as data:
            As = []
            error = []

            for line in data:
                row = line.split(',')
                As.append(float(row[0]))
                error.append(float(row[1]))

        As = np.array(As)
        error = np.array(error)

        fig, ax = plt.subplots()
        ax.errorbar(names, As, yerr=error, marker='o', linestyle='dashed', capsize=2.5)
        ax.set_yscale('log')
        plt.title('Average A(q)')
        plt.ylabel('Average A(q)')
        # plt.xticks(rotation=45)
        plt.xlabel(' ')
        plt.show()

    def directory_average_velocity(self):
        # Function to plot the average velocity of files in a directory
        filenames = self.filenames
        names = []

        f = open('average_vel.csv', 'w')
        f.close()

        dcif = DCIF(filename=filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=False)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()
        p0 = self.get_initial_conditions_swim()

        for i in range(len(filenames)):
            print(filenames[i])
            # print(self.dqs[i])
            dcif = DCIF(filename=filenames[i], error_name=self.stErr[i], dq=self.dqs[i], qloc=self.qloc, show=False)
            names.append(dcif.get_name())
            dcif.plot_Vs(qmax=max_q, qmin=min_q, show_choice=False, final_show=False, p0=p0)

        with open('average_vel.csv', 'r') as data:
            vels = []
            error = []

            for line in data:
                row = line.split(',')
                vels.append(float(row[0]))
                error.append(float(row[1]))

        vels = np.array(vels)
        error = np.array(error)

        fig, ax = plt.subplots()
        ax.errorbar(names, vels, yerr=error, marker='o', linestyle='dashed', capsize=2.5)
        ax.set_yscale('log')
        plt.title('Average Velocity')
        plt.ylabel('Average Velocity')
        # plt.xticks(rotation=45)
        plt.xlabel(' ')
        plt.show()

    def directory_average_tau0(self):
        # Function to plot the average tau0 of files in a directory
        filenames = self.filenames
        names = []

        f = open('tau0.csv', 'w')
        f.close()

        dcif = DCIF(filename=filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=False)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()
        p0 = self.get_initial_conditions_stretch()

        for i in range(len(filenames)):
            # print(filenames[i])
            dcif = DCIF(filename=filenames[i], error_name=self.stErr[i], dq=self.dqs[i], qloc=self.qloc, show=False)
            names.append(dcif.get_name())
            dcif.plot_tau0(qmax=max_q, qmin=min_q, show_choice=False, final_choice=False, p0=p0)

        with open('tau0.csv', 'r') as data:
            tau0_array = []
            error = []

            for line in data:
                row = line.split(',')
                tau0_array.append(float(row[0]))
                error.append(float(row[1]))

        tau0_array = np.array(tau0_array)
        error = np.array(error)

        print(tau0_array)
        print(error)
        print(names)

        fig, ax = plt.subplots()
        ax.errorbar(names, tau0_array, yerr=error, marker='o', linestyle='dashed', capsize=2.5)
        plt.title('Average tau0  for Tilapia Fish over a Range of Time', fontsize=20)
        plt.ylabel('Average tau_0', fontsize=18)
        plt.xticks(rotation=35)
        plt.xlabel('Time of Day')
        plt.show()

    def directory_average_beta(self):
        # Function to plot the average stretching exponent of files in a directory
        filenames = self.filenames
        names = []

        f = open('betas.csv', 'w')
        f.close()

        dcif = DCIF(filename=filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=False)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()
        p0 = self.get_initial_conditions_stretch()

        for i in range(len(filenames)):
            dcif = DCIF(filename=filenames[i], error_name=self.stErr[i], dq=self.dqs[i], qloc=self.qloc, show=False)
            names.append(dcif.get_name())
            dcif.plot_beta(qmax=max_q, qmin=min_q, show_choice=False, final_choice=False, p0=p0)

        with open('betas.csv', 'r') as data:
            beta_array = []
            error = []

            for line in data:
                row = line.split(',')
                beta_array.append(float(row[0]))
                error.append(float(row[1]))

        beta_array = np.array(beta_array)
        error = np.array(error)

        print(beta_array)
        print(error)
        print(names)

        fig, ax = plt.subplots()
        ax.errorbar(names, beta_array, yerr=error, marker='o', linestyle='dashed', capsize=2.5)
        plt.xticks(rotation=35)
        plt.title('Average Stretching Exponent for Tilapia Fish over a Range of Time')
        plt.ylabel('Average beta')
        plt.xlabel('Time')
        plt.show()

    def directory_average_diff(self):
        # Function to plot the average diffusion coefficient of files in a directory. Note filenames are sorted, so if
        # files are labelled by time, they will be plotted in chronological order,

        filenames = self.filenames
        names = []

        f = open('diff_coeffs.csv', 'w')
        f.close()

        dcif = DCIF(filename=filenames[0], error_name=self.stErr[0], dq=self.dqs[0], qloc=self.qloc, show=False)

        max_q = self.max_q_choice(dcif)
        min_q = self.min_q_choice()
        p0 = self.get_initial_conditions_swim()
        fit = self.fitting_algorithm_choice()

        for i in range(len(filenames)):
            # print(filenames[i])
            dcif = DCIF(filename=filenames[i], error_name=self.stErr[i], dq=self.dqs[i], qloc=self.qloc, show=False)
            names.append(dcif.get_name())
            dcif.plot_Ds(qmax=max_q, qmin=min_q, show_choice=False, fit=fit, final_choice=False, p0=p0)

        with open('diff_coeffs.csv', 'r') as data:
            diff = []
            error = []

            for line in data:
                row = line.split(',')
                diff.append(float(row[0]))
                error.append(float(row[1]))

        diff = np.array(diff)
        error = np.array(error)

        fig, ax = plt.subplots()
        ax.errorbar(names, diff, yerr=error, marker='o', linestyle='dashed', capsize=2.5)
        plt.title('Average Diffusion Coefficient')
        plt.ylabel('Average Diffusion Coefficient')
        plt.xticks(rotation=45)
        plt.xlabel(' ')
        plt.show()


main = Main()
main.run()
