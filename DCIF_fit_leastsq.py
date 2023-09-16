import matplotlib.pyplot as plt
import numpy as np
import pandas as pnda
import scipy.optimize as scopt
import seaborn as sns

sns.set_theme(style='ticks', palette='deep')


class DCIF:
    """
    Class takes in a text file produced by university DDM software and uses scipy to fit the data and return constants.

    WIP
    """

    def __init__(self, filename, error_name, dq, qloc, show):
        """
        :param filename: name of text file produced by DDM software containing raw DDM spectrum data
        :param error_name: name of text file produced by DDM software containing standard error on each raw data point
        :param dq: step of q values, found in JSON file produced by DDM software
        :param qloc: specific q value you wish to analyse/fit
        """

        self.tau0 = None
        self.beta = None
        self.A = None
        self.fit_times = None
        self.fit_data = None
        self.alpha = None
        self.D = None
        self.vel = None
        self.dq = dq
        self.filename = filename
        self.errorname = error_name
        self.df = pnda.read_csv(filename, sep='\t', header=None)
        self.err_df = pnda.read_csv(error_name, sep='\t', header=None)
        self.qloc = qloc
        self.show = show

    def show_data(self):
        # Function to show the raw data.
        data = self.df

        tau = np.array(data.iloc[:, 0])
        g = self.get_gs(self.qloc)

        errors = self.get_errors(self.qloc)

        fig, ax = plt.subplots()
        ax.errorbar(tau, g, errors)
        ax.set_xscale('log')
        plt.xlabel('Time')
        plt.ylabel('g(q, t)')
        plt.show()

    def get_name(self):
        # Takes the directory name from the overall path
        filename = self.filename
        first_slash_index = filename.find("/", 1)
        second_slash_index = filename.find("/", first_slash_index + 1)
        third_slash_index = filename.find("/", second_slash_index + 1)

        return filename[second_slash_index + 1: third_slash_index]

    def get_gs(self, qloc):
        # returns an array of the DCIF at a specific q
        data = self.df

        return np.array(data.iloc[:, qloc])

    def get_errors(self, qloc):
        #  returns an array of the errors at a specific q
        err_data = self.err_df

        return np.array(err_data.iloc[:, qloc])

    def get_q(self, qloc):
        # Converts raw q value into index
        return qloc * self.dq

    def get_q_max(self):
        # Helper function to find the last index of the range of q values
        df = self.df

        return len(df.axes[1]) - 1

    def brownian(self, p, tau, y, err):
        # ISF for simple diffusive behaviour
        qloc = self.qloc
        q = self.get_q(qloc)
        A = p[0]
        B = p[1]
        D = p[2]

        model = A * (1 - np.exp(-1 * D * (q ** 2) * tau)) + B

        resid = (y - model) / err
        return resid

    @staticmethod
    def stretched_exp(p, tau, y, err):
        # ISF for a stretched exponential function
        A = p[0]
        B = p[1]
        tau0 = p[2]
        beta = p[3]

        model = A * (1 - np.exp(-(tau/tau0) ** beta)) + B

        resid = (y - model) / err
        return resid

    def swim_diff(self, p, tau, y, err):
        # ISF for a combination of straight swimmers and diffusing particles
        qloc = self.qloc
        q = self.get_q(qloc)

        A = p[0]
        B = p[1]
        D = p[2]
        v = p[3]
        alpha = p[4]
        Z = p[5]

        part_one = (1 - alpha) * np.exp(-1 * (q ** 2) * D * tau)
        part_two = alpha * np.exp(-1 * (q ** 2) * D * tau)
        part_three = (Z + 1) / (Z * q * v * tau)
        part_four = np.sin(Z * np.arctan(((q * v * tau) / (Z + 1)))) / (
                    (1 + (((q * v * tau) / (Z + 1)) ** 2)) ** (Z / 2))

        f = part_one + (part_two * part_three * part_four)

        model = A * (1 - f) + B

        resid = (y - model) / err

        return resid

    def fit_diff(self):
        # Function to use least-squares fitting to fit the diffusive ISF to the data
        data = self.df
        qloc = self.qloc

        data, fit_data, fit_err, fit_tau, tau = self.get_fit_variables(data, qloc)

        R = scopt.least_squares(self.brownian, [1, 0, .3], args=(fit_tau, fit_data, fit_err), method='trf')

        self.A = R.x[0]
        self.D = R.x[2]
        print(R.x)

        self.show_fit_diff(R.x, tau, data)

    def get_fit_variables(self, data, qloc):
        # Helper function to return various variables needed to fit the data

        #  The array of lag times
        tau = np.array(data.iloc[:, 0], dtype=float)
        self.fit_times = tau

        # The data and error given by DDM
        data = np.array(data.iloc[:, qloc], dtype=float)
        err = self.get_errors(qloc)

        # Apply the weighing to provide weighted arrays for fitting
        weightings = self.get_weightings()
        fit_tau = np.repeat(tau, weightings)
        fit_data = np.repeat(data, weightings)
        fit_err = np.repeat(err, weightings)

        return data, fit_data, fit_err, fit_tau, tau

    def fit_stretched_exp(self, p0):
        # Function to use least-squares fitting to fit the stretched exponential ISF to the data
        data = self.df
        qloc = self.qloc

        data, fit_data, fit_err, fit_tau, tau = self.get_fit_variables(data, qloc)

        R = scopt.least_squares(self.stretched_exp, p0, args=(fit_tau, fit_data, fit_err), method='trf')

        self.A = R.x[0]
        self.tau0 = R.x[2]
        self.beta = R.x[3]

        print(R.x)

        self.show_fit_stretch(R.x, tau, data)

    def fit_swim(self, p0):
        data = self.df
        qloc = self.qloc

        data, fit_data, fit_err, fit_tau, tau = self.get_fit_variables(data, qloc)

        R = scopt.least_squares(self.swim_diff, p0, args=(fit_tau, fit_data, fit_err), method='trf')

        self.A = R.x[0]
        self.D = R.x[2]
        self.vel = R.x[3]
        self.alpha = R.x[5]

        print(R.x)

        self.show_fit_swim(R.x, tau, data)

    def get_weightings(self):
        # Function to return an array of weightings - the inverse of the standard deviation at each point multiplied
        # by a sufficient integer

        qloc = self.qloc
        err = self.get_errors(qloc)

        weightings = 1/(np.sqrt(err)) * 600
        weightings = weightings.astype(int)
        return weightings

    def show_fit_diff(self, res, tau, data):
        # Function to display data and fitted curve, from the swimDiff model.
        qloc = self.qloc
        q = self.get_q(qloc)

        A = res[0]
        B = res[1]
        D = res[2]

        model = A * (1 - np.exp(-1 * D * (q ** 2) * tau)) + B

        self.fit_data = model

        if self.show:
            fig, ax = plt.subplots()
            ax.plot(tau, data, label='raw data', marker='o', markerfacecolor='none', ms=6, markeredgecolor='black',
                    linestyle='none')
            ax.plot(tau, model, color='red', label='fitted curve')
            ax.set_xscale('log')

            ax.legend()
            plt.xlabel('Time')
            plt.ylabel('g(q, t)')
            plt.title('DCIF with q = ' + str(self.qloc - 1))
            plt.show()

    def show_fit_stretch(self, res, tau, data):
        # Function to display data and fitted curve, from the stretched exponential model.
        A = res[0]
        B = res[1]
        tau0 = res[2]
        beta = res[3]

        model = A * (1 - np.exp(-(tau / tau0) ** beta)) + B
        self.fit_data = model

        if self.show:
            fig, ax = plt.subplots()
            ax.plot(tau, data, label='raw data', marker='o', markerfacecolor='none', ms=6, markeredgecolor='black',
                    linestyle='none')
            ax.plot(tau, model, color='red', label='fitted curve')
            ax.set_xscale('log')

            ax.legend()
            plt.xlabel('Time')
            plt.ylabel('g(q, t)')
            plt.title('DCIF with q = ' + str(self.qloc - 1))
            plt.show()

    def show_fit_swim(self, res, tau, data):
        # Function to display data and fitted curve, from the swimDiff model.
        qloc = self.qloc
        q = self.get_q(qloc)

        A = res[0]
        B = res[1]
        D = res[2]
        v = res[3]
        alpha = res[4]
        Z = res[5]

        part_one = (1 - alpha) * np.exp(-1 * (q ** 2) * D * tau)
        part_two = alpha * np.exp(-1 * (q ** 2) * D * tau)
        part_three = (Z + 1) / (Z * q * v * tau)
        part_four = np.sin(Z * np.arctan(((q * v * tau) / (Z + 1)))) / (
                (1 + (((q * v * tau) / (Z + 1)) ** 2)) ** (Z / 2))

        f = part_one + (part_two * part_three * part_four)

        model = A * (1 - f) + B

        self.fit_data = model

        if self.show:
            fig, ax = plt.subplots()
            ax.plot(tau, data, label='raw data', marker='o', markerfacecolor='none', ms=6, markeredgecolor='black',
                    linestyle='none')
            ax.plot(tau, model, color='red', label='fitted curve')
            ax.set_xscale('log')

            ax.legend()
            plt.xlabel('Time')
            plt.ylabel('g(q, t)')
            plt.title('DCIF with q = ' + str(self.qloc - 1))
            plt.show()

    def plot_Vs(self, qmax, qmin, show_choice, final_show, p0):
        # Function writes the average velocity and error to file, then plots the velocity for each q value.
        qlocs = np.arange(qmin, qmax)
        Vs = []

        Vs, av_v, err = self.calc_Vs(Vs, qlocs, show_choice, p0)

        f = open('average_vel.csv', 'a')
        f.write('%lf,%lf\n' % (float(av_v), err))
        f.close()

        if final_show:
            name = self.get_name()

            plt.scatter(qlocs, Vs, color='black')
            plt.hlines(av_v, 0, qlocs[-1] + 2, colors='red', label='average', linewidth=err, alpha=0.5)
            plt.xlim(0, qmax + 1)
            plt.legend()
            plt.title('Velocity vs q for ' + str(name))
            plt.xlabel('q ($\mu m^{-1}$)')
            plt.ylabel('Velocity')
            plt.show()

    def calc_Vs(self, Vs, qlocs, show_choice, p0):
        # Function to calculate the average velocity and error for a given file
        for q in qlocs:
            dcif = DCIF(self.filename, self.errorname, self.dq, qloc=q, show=show_choice)

            dcif.fit_swim(p0)  # Make sure show_fit is commented out

            print(q)
            Vs.append(dcif.vel)

        Vs = np.array(Vs)
        av_v = np.mean(Vs)
        err = np.std(Vs) / np.sqrt(len(Vs))

        print('The average velocity is: ' + str(av_v) + ' ± ' + str(err))

        return Vs, av_v, err

    def plot_As(self, qmax, qmin, show_choice, fit, final_choice, p0):
        # Function writes the average A and error to file, then plots the value of A for each q value.
        qlocs = np.arange(qmin, qmax)
        As = []

        if fit == 1:
            for q in qlocs:
                dcif = DCIF(self.filename, self.errorname, self.dq, qloc=q, show=show_choice)
                dcif.fit_diff()

                print(q)
                As.append(dcif.A)

        elif fit == 2:
            for q in qlocs:
                dcif = DCIF(self.filename, self.errorname, self.dq, qloc=q, show=show_choice)
                dcif.fit_swim(p0)

                print(q)
                As.append(dcif.A)

        elif fit == 3:
            for q in qlocs:
                dcif = DCIF(self.filename, self.errorname, self.dq, qloc=q, show=show_choice)
                dcif.fit_stretched_exp(p0)

                print(q)
                As.append(dcif.A)
        else:
            print('This should not have happened')

        As = np.array(As)
        av_A = np.mean(As)
        err = np.std(As) / np.sqrt(len(As))

        print('The value for A  is: ' + str(av_A) + ' ± ' + str(err))

        f = open('const_A.csv', 'a')
        f.write('%lf,%lf\n' % (float(av_A), err))
        f.close()

        if final_choice:
            name = self.get_name()
            plt.scatter(qlocs, As, color='black')
            plt.hlines(av_A, qmin, qlocs[-1] + 2, colors='red', label='average', linewidth=err, alpha=0.5)
            plt.legend()
            plt.title('A(q) vs q for ' + str(name))
            plt.xlabel('q ($\mu m^{-1}$)')
            plt.ylabel('A(q)')
            plt.show()

    def plot_Ds(self, qmax, qmin, show_choice, fit, final_choice, p0):
        # Function writes the average D and error to file, then plots the value of D for each q value.

        qlocs = np.arange(qmin, qmax)
        Ds = []

        if fit == 1:
            for q in qlocs:
                dcif = DCIF(self.filename, self.errorname, self.dq, qloc=q, show=show_choice)
                dcif.fit_diff()

                print(q)
                Ds.append(dcif.D)

        elif fit == 2:
            for q in qlocs:
                dcif = DCIF(self.filename, self.errorname, self.dq, qloc=q, show=show_choice)
                dcif.fit_swim(p0)

                print(q)
                Ds.append(dcif.D)
        else:
            print('This should not have happened')

        Ds = np.array(Ds)
        av_D = np.mean(Ds)
        err = np.std(Ds) / np.sqrt(len(Ds))

        print('The value for D  is: ' + str(av_D) + ' ± ' + str(err))

        f = open('diff_coeffs.csv', 'a')
        f.write('%lf,%lf\n' % (float(av_D), err))
        f.close()

        if final_choice:
            plt.scatter(qlocs, Ds, color='black')
            plt.hlines(av_D, qmin, qlocs[-1] + 2, colors='red', label='average', linewidth=err, alpha=0.5)
            plt.legend()
            plt.title('Diffusion Coefficient vs q')
            plt.xlabel('q ($\mu m^{-1}$)')
            plt.ylabel('D ($\mu m^2 /s$)')
            plt.show()

    def plot_alpha(self, p0):
        # Function writes the average alpha and error to file, then plots the alpha for each q value.

        qmax = self.get_q_max()
        qlocs = np.arange(2, qmax)
        alphas = []

        for q in qlocs:
            dcif = DCIF(self.filename, self.errorname, self.dq, qloc=q, show=False)
            dcif.fit_swim(p0)  # Make sure show_fit is commented out

            print(q)
            alphas.append(dcif.alpha)

        alphas = np.array(alphas)
        av_alpha = np.mean(alphas)
        err = np.std(alphas) / np.sqrt(len(alphas))

        print('The average value for alpha  is: ' + str(av_alpha) + ' ± ' + str(err))

        plt.scatter(qlocs, alphas, color='black')
        plt.hlines(av_alpha, 0, qlocs[-1] + 2, colors='red', label='average', linewidth=err, alpha=0.5)
        plt.legend()
        plt.title('Alpha vs q')
        plt.xlabel('q ($\mu m^{-1}$)')
        plt.ylabel('Alpha)')
        plt.show()

    def plot_beta(self, qmax, qmin, show_choice, final_choice, p0):
        # Function writes the average beta and error to file, then plots the beta for each q value.
        qlocs = np.arange(qmin, qmax)
        betas = []

        for q in qlocs:
            dcif = DCIF(self.filename, self.errorname, self.dq, qloc=q, show=show_choice)
            dcif.fit_stretched_exp(p0)

            print(q)
            betas.append(dcif.beta)

        betas = np.array(betas)
        av_beta = np.mean(betas)
        err = np.std(betas) / np.sqrt(len(betas))

        print('The average value for beta  is: ' + str(av_beta) + ' ± ' + str(err))

        f = open('betas.csv', 'a')
        f.write('%lf,%lf\n' % (float(av_beta), err))
        f.close()

        if final_choice:
            name = self.get_name()
            plt.scatter(qlocs, betas, color='black')
            plt.hlines(av_beta, qmin, qlocs[-1] + 2, colors='red', label='average', alpha=0.5)
            plt.legend()
            plt.title('beta vs q for ' + str(name))
            plt.xlabel('q ($\mu m^{-1}$)')
            plt.ylabel('beta(q)')
            plt.show()

    def plot_tau0(self, qmax, qmin, show_choice, final_choice, p0):
        # Function writes the average tau0 and error to file, then plots the tau0 for each q value.
        qlocs = np.arange(qmin, qmax)
        tau0_array = []

        for q in qlocs:
            dcif = DCIF(self.filename, self.errorname, self.dq, qloc=q, show=show_choice)
            dcif.fit_stretched_exp(p0)

            print(q)
            tau0_array.append(dcif.tau0)

        tau0_array = np.array(tau0_array)
        av_tau0 = np.mean(tau0_array)
        err = np.std(tau0_array) / np.sqrt(len(tau0_array))

        print('The average value for tau0  is: ' + str(av_tau0) + '±' + str(err))

        f = open('tau0.csv', 'a')
        f.write('%lf,%lf\n' % (float(av_tau0), err))
        f.close()

        if final_choice:
            name = self.get_name()
            fig, ax = plt.subplots()
            ax.scatter(qlocs, tau0_array, color='black')
            ax.hlines(av_tau0, qmin, qlocs[-1] + 2, colors='red', label='average', alpha=0.5)
            plt.legend()
            plt.title('tau0 vs q for ' + str(name))
            plt.xlabel('q ($\mu m^{-1}$)')
            plt.ylabel('tau0(q)')
            plt.show()
