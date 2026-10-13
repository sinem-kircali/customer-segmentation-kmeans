# Customer Segmentation with k-means: Hotel Guests, App Users, and When k-means Misleads

This project covers two segmentation case studies in one R Markdown analysis:
- **Hotel guests** (behavioural data): which guest segments generate the revenue, and how should the hotel target them?
- **Social-app users** (mixed demographic data): what user types exist? This case shows how k-means on one-hot encoded categories can produce misleading segments, and compares it with Gower distance + PAM.

Built for the *Strategic Marketing Decision Making* course (hotel case, group assignment) and the *Seminar Data Science for Marketing Analytics* course (app-user case, individual assignment), MSc Data Science & Marketing Analytics, Erasmus University Rotterdam.

**→ Full analysis with code, tables and figures: [customer_segmentation.md](customer_segmentation.md)**

## Key results
**Hotel guests** (9,384 customers, 9 behavioural variables, k-means with k = 4):

| Segment | Guests | Lodging revenue share | Ancillary revenue share | Profile |
|---|---|---|---|---|
| High-value long stays | 11% | **53%** | **54%** | ~6.5 room nights, groups/families, book ~110 days ahead |
| Older early planners | 16% | 14% | 20% | age ~66, book ~250 days ahead, highest ancillary-to-lodging ratio |
| Short, low-spend stays (two segments) | 73% | 33% | 27% | differ mainly in customer tenure |

Recommendations:
- **High-value long stays:** retention and pre-arrival upselling.
- **Older early planners:** early-booking packages with F&B or spa credit.

Robustness check: on log-transformed revenue the segments change a lot (adjusted Rand index 0.27). The high-value segment is the long tail of big spenders, so its boundaries depend on how revenue is scaled.

**App users** (1,113 users, mixed data):
- **k-means on standardised one-hot dummies:** each segment copies one dummy column. One segment is the highly educated users; the other two are exact 100% female and 100% male splits.
- **Gower distance + PAM:** finds a softer two-segment structure, mostly married women vs mostly single men. This one is real: in this sample 79% of women are married vs 24% of men.
- **Lesson:** check which variables drive each segment before naming it, and use a mixed-data distance for demographic data.

## Methods
- Standardisation; elbow and silhouette checks (on a 3,000-customer subsample for the silhouette); `nstart = 25` random starts.
- The share of each variable's variance explained by the segments (between-cluster SS / total SS), used to see what defines the segments.
- Average vs **total** revenue contribution per segment, for targeting.
- Robustness: log-transformed re-run, compared with the adjusted Rand index.
- Mixed data: Gower distance (`cluster::daisy`) + PAM, with k chosen by silhouette and medoids reported as representative users.

## Notes on this version
Merged and extended from two separate assignments:
- k-means now uses multiple random starts (the hotel script had one);
- profiles report income on its original scale (it had been summarised after the log transform);
- the revenue-share table, the variable-driver analysis, the log-transform robustness check and the Gower + PAM comparison are new.

## Structure
```
customer_segmentation.Rmd    source (R Markdown)
customer_segmentation.md     rendered analysis, viewable on GitHub
customer_segmentation_files/ figures
data/                        place the data files here (not included)
```
The hotel data comes from Antonio, N., de Almeida, A. & Nunes, L. (2020), *A hotel's customers personal, behavioral, demographic, and geographic dataset from Lisbon, Portugal (2015–2018)*, Data in Brief 33. The analysis uses a 10,001-customer sample (`data/HotelCustomersSubset_sample.csv`). The app-user data (`data/CustomerSegmentation.RData`) was provided by the course.

## Tech stack
R · tidyverse · cluster (Gower, PAM) · factoextra · R Markdown
