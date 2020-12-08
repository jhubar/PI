import matplotlib.pyplot as plt

def plot_age(x,y):
    plt.plot(x, y, c='red', label = 'age')
    plt.legend()
    plt.savefig('Plot/age.png')
    plt.close()

def plot_age2(x,y):
    plt.plot(x, y, c='red', label = 'age')
    plt.legend()
    plt.savefig('Plot/age_smooth.png')
    plt.close()

def plot_households(category,households_number):
    plt.plot(category,households_number, label = 'households')
    plt.legend()
    plt.savefig('Plot/households.png')
    plt.close()

def plot_households_smooth(category,households_number):
    plt.plot(category,households_number, label = 'households')
    plt.legend()
    plt.savefig('Plot/households_smooth.png')
    plt.close()

def plot_workplaces(category,number):
    plt.bar(category,number, label = 'workplaces')
    plt.legend()
    plt.savefig('Plot/workplaces.png')
    plt.close()

def plot_medium_workplaces(x,y,mean):
    plt.bar(x,y, label = 'Number of medium workplaces')
    plt.axhline(mean, c= 'red',label = 'mean :'+str(round(mean,2)))
    plt.legend()
    plt.savefig('Plot/medium_workplaces.png')
    plt.close()

def plot_large_workplaces(x,y,mean):
    plt.bar(x,y, label = 'Number of medium workplaces')
    plt.axhline(mean, c= 'red',label = 'mean :'+str(round(mean,2)))
    plt.legend()
    plt.savefig('Plot/large_workplaces.png')
    plt.close()

def plot_schools(category,number):
    plt.bar(category,number, label = 'schools')
    plt.legend()
    plt.savefig('Plot/schools.png')
    plt.close()

def plot_small_schools(x,y,mean):
    plt.bar(x,y, label = 'Number of small schools')
    plt.axhline(mean, c= 'red',label = 'mean :'+str(round(mean,2)))
    plt.legend()
    plt.savefig('Plot/small_schools.png')
    plt.close()

def plot_medium_schools(x,y,mean):
    plt.bar(x,y, label = 'Number of medium schools')
    plt.axhline(mean, c= 'red',label = 'mean :'+str(round(mean,2)))
    plt.legend()
    plt.savefig('Plot/medium_schools.png')
    plt.close()

def plot_large_schools(x,y,mean):
    plt.bar(x,y, label = 'Number of medium schools')
    plt.axhline(mean, c= 'red',label = 'mean :'+str(round(mean,2)))
    plt.legend()
    plt.savefig('Plot/large_schools.png')
    plt.close()
