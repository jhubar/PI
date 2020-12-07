import matplotlib.pyplot as plt

def plot_age(x,y):
    plt.plot(x, y, c='red', label = 'age')
    plt.legend()
    plt.savefig('Plot/age.png')
    plt.close()

def plot_households(category,households_number):
    plt.bar(category,households_number, label = 'households')
    plt.legend()
    plt.savefig('Plot/households.png')
    plt.close()

def plot_workplaces(x,y):
    plt.plot(x, y, c='red', label = 'workplaces')
    plt.legend()
    plt.savefig('Plot/workplaces.png')
    plt.close()

def plot_schools(x,y):
    plt.plot(x, y, c='red', label = 'schools')
    plt.legend()
    plt.savefig('Plot/schools.png')
    plt.close()
