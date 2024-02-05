from plotnine import *
import pandas as pd

time = "2024-01-24_20:40"
time = "2024-01-30_14:24"
time = "2024-01-30_15:31" # classifier
time = "2024-01-31_11:30" # classifier_reg
time = "2024-02-04_19:53" # classifier_reg
time = "2024-02-04_22:36" # classifier_reg
time = "2024-02-05_10:50" # reg

df = pd.read_csv(f"/projects/genomic-ml/da2343/ml_project_1/reproduce/results/{time}_results.csv")
df = df.drop(columns=['Index of Predicted Column'])
df = df.groupby(['Dataset', 'Algorithm', 'Predicted Column Name']).mean().reset_index()


p = ggplot(df)
p = p + geom_point(aes(x="Mean Squared Error", y="Dataset", color="Algorithm"))
p = p + facet_grid("~Predicted Column Name", scales="free")
p = p + labs(
    x="MSE", y="Dataset"
)
p = p + theme(axis_text_x=element_text(angle=90))
p = p + theme(figure_size=(1000, 5))
p = p + scale_x_log10()

# save plot
p.save(f"{time}_results.png", limitsize=False)