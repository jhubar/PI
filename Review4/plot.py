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
    plt.bar(category,households_number, label = 'households')
    plt.legend()
    plt.savefig('Plot/households.png')
    plt.close()

def plot_communities(category,communities_number):
    plt.bar(category,communities_number, label = 'communities')
    plt.legend()
    plt.savefig('Plot/communities.png')
    plt.close()

def plot_pie_households():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = '1', '2', '3','4','5','6','7'
    sizes = [125898/394019, 108977/394019, 62988/394019, 47849/394019, 24993/394019, 11855/394019, 11431/394019]
    explode = (0, 0, 0, 0, 0 ,0.1,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('Plot/pie_households.png')
    plt.close()

def plot_pie_communities():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = '1', '2', '3','4','5','6','7'
    sizes = [125898/394019, 108977/394019, 62988/394019, 47849/394019, 24993/394019, 11855/394019, 11431/394019]
    explode = (0, 0, 0, 0, 0 ,0.1,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('Plot/pie_communities.png')
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

def plot_pir():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'young', 'junior', 'medior', 'seignior'
    sizes = [75211/1000324,216665/1000324 , 586837/1000324,121611/1000324]
    explode = (0, 0, 0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('Plot/pie_dist.png')
    plt.close()

def plot_pie_schools():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'small: [0-10]', 'medium: [100-500]', 'large: [500+]'
    sizes = [61/545,416/545,68/545]
    explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('Plot/pie_schools.png')
    plt.close()

def plot_pie_workplaces():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'small: [0-10]', 'medium: [100-500]', 'large: [500+]'
    sizes = [18208/24347,5333/24347,806/24347]
    explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('Plot/pie_workplaces.png')
    plt.close()
