# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 14:44:10 2023

@author: DELL
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(file_paths, selected_countries, start_year, end_year):
    
    
    """
    Read and preprocess data from CSV files.

    Parameters:
    - file_paths (list of str): List of file paths for multiple datasets.
    - selected_countries (list of str): List of countries to include in the analysis.
    - start_year (int): Start year for data extraction.
    - end_year (int): End year for data extraction.

    Returns:
    - dataframes_dict (dict): Dictionary of DataFrames with keys as file names.
    - dataframes_dict_transpose (dict): Dictionary of transposed DataFrames.
    """
     
    # Dictionary to store original DataFrames
    dataframes_dict = {}
    # Dictionary to store transposed DataFrames
    dataframes_dict_transpose = {}
    
    # Columns to exclude
    exclude_columns = ['Country Code', 'Indicator Name', 'Indicator Code']
    
    
    # Iterate over each file path
    for path in file_paths:
        
        # Extract file name without extension
        file_name = path.split('.')[0].replace(' ', '_')
        
        # Load the dataset, skipping the first 4 rows
        df = pd.read_csv(path, skiprows=4, usecols=lambda x: x.strip() != "Unnamed: 67" if x not in ['Indicator Name', 'Indicator Code', 'Country Code'] else True)
        
        # Exclude specified columns
        df = df.drop(columns=exclude_columns, errors='ignore')
        
        # Set 'Country Name' as the index
        df.set_index("Country Name", inplace=True)
        
        # Filter data for selected countries and the specified year range
        df = df.loc[selected_countries, "2012":"2022"]
        
        # Calculate the mean of each row
        df['mean'] = df.mean(axis=1)
        
        # Fill null values in each row with the mean of that row
        df = df.apply(lambda row: row.fillna(row['mean']), axis=1)
        
        # Remove the 'mean' column from the dictionary
        df = df.drop(columns=['mean'])
        
        # Transpose the DataFrame
        df_trans = df.transpose()
        
        df_trans.dropna(axis=0, how="all", inplace=True)
        
        # Reset index to make years a column
        df_trans.reset_index(inplace=True)
        df_trans = df_trans.rename(columns={'index': "Year"})
        
        # Convert 'Year' column to integers
        df_trans['Year'] = df_trans['Year'].astype(int)
        
        # Store DataFrames in dictionaries
        dataframes_dict[file_name] = df
        dataframes_dict_transpose[file_name] = df_trans
        
        
    return dataframes_dict, dataframes_dict_transpose


def create_line_plot(data_frame, title):
    
    
    
    #Create and display a line plot for the countries.

    
    # Set the figure size
    plt.figure(figsize=(12, 6))
    
    output_file_path = f"{title}_lineplot.png"
    # Iterate over each country column (excluding the 'Year' column)
    for country in data_frame.columns[1:]:
        
        
        # Calculate the percentage increase compared to the initial year
        initial_value = data_frame[country].iloc[0]
        final_value = data_frame[country].iloc[-1]
        percentage_change = ((final_value - initial_value) / initial_value) * 100
        plt.plot(data_frame['Year'], data_frame[country], label=f"{country} {percentage_change:+.1f}%")
        
        # Plot the percentage increase
        #plt.plot(data_frame['Year'], percentage_increase, label=country)

        
    
    # Set labels and title
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.title(f"{title} Over Years", fontsize=14)
    plt.xticks(fontsize=14)
    
    # Add a grid for better readability
    plt.grid(True)
    
    # Add a legend with the country names
    plt.legend(title='Country', bbox_to_anchor=(1, 1), fontsize=14)
    
    # Adjust layout for better visualization
    plt.tight_layout()
    # Save the image if the output_file_path is provided
    if output_file_path:
       plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
    
    # Display the plot
    #plt.show()
    
    
def create_line_plot_for_country(data_frame, countries, title_name):
    
    
    """
    Create and display a line plot for a specific country.

    Parameters:
    - data_frame (pd.DataFrame): The DataFrame containing the data.
    - country_name (str): The name of the country for which to create the plot.
    - title (str): The title of the plot.

    Returns:
    - None
    
    """
   
    for country in countries:
        
        if country in data_frame.columns:
            # Set the figure size
            plt.figure(figsize=(12, 6))

            # Plot the data for the selected country
            plt.plot(data_frame['Year'], data_frame[country], label=country)

            # Set labels and title
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Values', fontsize=14)
            plt.title(f"{title_name} Over Years for {country}", fontsize=14)
            plt.xticks(fontsize=14)

            # Add a grid for better readability
            plt.grid(True)

            # Add a legend with the country name
            plt.legend(fontsize=14)

            # Adjust layout for better visualization
            plt.tight_layout()
            
            output_file_path = f"{title_name}_{country}_linePlot.png"
            
            if output_file_path:
               plt.savefig(output_file_path, bbox_inches='tight', dpi=300)

            # Display the plot
            #plt.show()


def create_bar_chart(data_frame, y_label, title):
    
    
    """
    Create and display a bar chart.

    Args:
        data_frame (pd.DataFrame): Data for plotting.
        y_label (str): Label for the y-axis.
        title (str): Title of the chart.
   """
    output_file_path = f"{title}_barchart.png"
    # Set 'Year' as the index
    data_frame.set_index("Year", inplace=True)

    # Plotting bar chart
    data_frame.plot(kind="bar", figsize=(10, 6))
    
    # Set labels and title
    plt.xlabel("Year", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title)
    plt.xticks(fontsize=14)
    
    # Add legend with DataFrame names
    plt.legend(title="Country", bbox_to_anchor=(1, 1) , fontsize=14)
    
    if output_file_path:
       plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
    
    # Display the plot
    #plt.show()


def create_2022_bar_chart(dataframes):
    """
    Create and display a bar chart.

    Args:
        data_frame (pd.DataFrame): Data for plotting.
        y_label (str): Label for the y-axis.
        title (str): Title of the chart.
    """
    
    num_rows = len(dataframes) // 2 + len(dataframes) % 2  # Calculate the number of rows for subplots
    num_cols = 2  # Fixed number of columns for subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 10))
    fig.suptitle("2022 Data Overview", fontsize=16)

    for i, (df_name, df) in enumerate(dataframes.items()):
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axes[row_idx, col_idx]

        if 'Year' in df.columns:
            df.set_index("Year", inplace=True)

        df.loc[2022].plot(kind="bar", ax=ax, color='royalblue', edgecolor='black')

        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=14)

        # Remove the x-axis label "Country" if not needed
        ax.set_xlabel('')

        # Use ylabel to put the indicator name to the left of the bars, if that's the desired location
        ax.set_ylabel(df_name.title(), fontsize=14)
        
        # Only set the title if you want it above the bars; otherwise, comment this out
        # ax.set_title(df_name.title(), fontsize=14)
        
        ax.tick_params(labelsize=14)

        # Hide x-axis labels for all but the bottom row subplots
        if row_idx < num_rows - 1:
            ax.set_xticklabels([])
        else:
            # Only rotate x-ticks for the bottom subplots
            ax.tick_params(axis='x', rotation=45)

    # Remove any empty subplots
    for i in range(len(dataframes), num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    plt.savefig("2022 Data Overview", bbox_inches='tight', dpi=300)
    
    #plt.show()

def create_pie_chart(mean_by_country, title, output_file_path):
    """
    Create and display a pie chart for a specific year.

    Args:
        data_frame (pd.DataFrame): Data for plotting.
        year (int): Year for which the pie chart is to be created.
        title (str): Title of the pie chart.
        output_file_path (str, optional): File path to save the image. If None, the image will not be saved.
    """
    #output_file_path = f"{title}_piechart.png"
    
    plt.pie(mean_by_country.values, labels=mean_by_country.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    #plt.title('Distribution of Mean Inflation by Country')
    
    # Display the chart
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # Setting the font size for the autopct
    plt.rcParams['font.size'] = 14
    # Save the image if the output_file_path is provided
    if output_file_path:
       plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
       print(output_file_path)
    plt.show()
    
    
def create_2022_pie_chart(data_frame, title):
    """
    Create and display a pie chart for the year 2022.

    Args:
        data_frame (pd.DataFrame): Data for plotting.
        title (str): Title of the chart.
    """
    output_file_path = f"{title}_piechart.png"
    # Ensure 'Year' is the index for plotting
    if 'Year' in data_frame.columns:
        data_frame = data_frame.set_index("Year")
    # Select the data for the year 2022
    data_2022 = data_frame.loc[2022]

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(data_2022, labels=data_2022.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal') 
    # Setting the font size for the autopct
    plt.rcParams['font.size'] = 14

    # Show the plot
    plt.tight_layout()
    
    if output_file_path:
       plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
       
    #plt.show()




if __name__ == "__main__":
    file_paths = ['urban population.csv','rural population.csv','inflation.csv','GDP Growth.csv','Fuel imports.csv','Trade.csv']
    selected_countries = ['Canada', 'China', 'Germany', 'United Kingdom', 'India', 'United States']
    start_year = 2012
    end_year = 2022
    # Call the read function
    dataframe, dataframes_dict_transpose = read_data(file_paths, selected_countries, start_year, end_year)
    
    #create_bar_chart(dataframes_dict_transpose, df_names, selected_countries)
    df_names = ['urban_population','rural_population','inflation','GDP_Growth','Fuel_imports','Trade',]
    dataframes = {
        "inflation" : dataframes_dict_transpose["inflation"],
        "GDP_Growth" : dataframes_dict_transpose["GDP_Growth"],
        "urban_population" : dataframes_dict_transpose["urban_population"],
        "rural_population" : dataframes_dict_transpose["rural_population"],
        "Fuel_imports" : dataframes_dict_transpose["Fuel_imports"],
        "Trade" : dataframes_dict_transpose["Trade"]
        
    }
    
    #call line plot function according to categories
    for df_name, df in dataframes.items():
        title_name = df_name.replace("_", " ").title()  # Create a y-axis label from DataFrame name
        create_line_plot(df, title_name)
    
    #call line plot function for country
    for df_name, df in dataframes.items():
        title_name = df_name.replace("_", " ").title()  # Create a y-axis label from DataFrame name
        create_line_plot_for_country(df, selected_countries, title_name)
    
    for df_name, df in dataframes.items():
        y_label = df_name.replace("_", " ").title()  # Create a y-axis label from DataFrame name
        title = f"{y_label} Over Years"
        #print(df)
        create_bar_chart(df, y_label, title)
        
    for df_name, df in dataframes.items():
        y_label = df_name.replace("_", " ").title()  # Create a y-axis label from DataFrame name
        title = f"{y_label} in 2022"
        create_2022_bar_chart(dataframes)
        
    for df_name, df in dataframes.items():
        mean_by_country = df.iloc[:,1:].mean(axis=0)
        y_label = df_name.replace("_", " ").title()  # Create a y-axis label from DataFrame name
        title = f"Distribution of {y_label} Over Years"
        output_file_path = f"{df_name}_PieChart.png"
        #print(df_name)
        create_pie_chart(mean_by_country, title, output_file_path)
    
    for df_name, df in dataframes.items():
        y_label = df_name.replace("_", " ").title()  # Create a y-axis label from DataFrame name
        title = f"{y_label} in 2022"
        create_2022_pie_chart(df, title)
    
    
    # Create a new figure for the composite image
    final_image_graphs = plt.figure(figsize=(16, 12))
    
    # List of individual plot file paths
    image_paths = ['Inflation_lineplot.png', 'Gdp Growth Over Years_barchart.png', 'Fuel_imports_PieChart.png',
                   'Trade_linePlot.png', 'urban_population_PieChart.png','2022 Data Overview.png']
    image_paths1 = ['Fuel_imports_PieChart.png',
                    'urban_population_PieChart.png']
    
    # Specify the layout for subplots
    num_rows = 2
    num_cols = 3
    
    # Loop through the individual plot file paths and add them as subplots to the composite image
    for i, image_path in enumerate(image_paths):
        plt.subplot(num_rows, num_cols, i + 1)
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis('off')  # Turn off axes for individual plots
    
    # Adjust spacing between subplots;/nb 
    plt.tight_layout()
    
    # Save the composite image
    final_image_graphs.savefig('final_image_graphs.png',bbox_inches='tight', dpi=300)
