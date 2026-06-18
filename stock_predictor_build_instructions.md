# Stock Return Predictor — Build-It-Yourself Guide (No Code)

> This guide tells you **what** to build, **why** it works that way, **what inputs/outputs** each piece needs, and **what mistakes to watch for**. No code. Write it yourself — refer to the companion code guide only if you get stuck.

How to use this: work top to bottom. Each step has a "Definition of Done" — don't move to the next step until you hit it. Each step also has a "Common Mistakes" list specific to that step; read it BEFORE you start coding, not after you're stuck.

---

## Step 0 — Project Skeleton

Before writing logic, set up the folder structure so you're not reorganizing later.

**What to create:**
- A `backend/` folder containing a `data/` subfolder and a `models/` subfolder
- A `frontend/` folder
- A `requirements.txt` at the project root or inside `backend/`
- A `.env` or config file for things like FRED API access (you may not need a key for `pandas_datareader`, but check)

**Definition of done:** You have empty files for: `fetcher.py`, `features.py`, `baseline.py`, `ml_pipeline.py`, `monte_carlo.py`, `dcf.py`, `main.py`, `app.py`. Nothing in them works yet — that's fine.

**Common mistakes:**
- Putting everything in one giant script. You'll want clean separation later when you build the API, because the API just imports functions from these files.

---

## Step 1 — Data Fetcher (Macro + Equity Premium)

### Goal
Produce one clean monthly DataFrame containing: the S&P 500 return, a risk-free rate, the equity premium (return minus risk-free rate), and the macro predictor variables from Welch & Goyal.

### What you need conceptually

**The equity premium** is the entire target variable for Tab 1. It is defined as:

```
equity premium = stock market return − risk-free rate
```

Why subtract the risk-free rate? Because a 10% stock return when risk-free bonds pay 8% is a very different story than a 10% return when bonds pay 1%. The "premium" isolates the *extra* compensation investors get for taking on stock risk. This is the quantity WG (and most asset-pricing research) actually try to predict — not the raw return.

**Data you need, and where it comes from:**

| Data | Source | Frequency you'll resample to |
|---|---|---|
| S&P 500 price | yfinance, ticker `^GSPC` | Monthly |
| Risk-free rate | FRED series `TB3MS` (3-month T-bill) via `pandas_datareader` | Monthly |
| 10-year Treasury yield | FRED `DGS10` | Monthly |
| BAA corporate bond yield | FRED `BAMLC0A4CBBB` (or similar BAA series) | Monthly |
| AAA corporate bond yield | FRED `BAMLC0A1CAAA` | Monthly |
| CPI (for inflation) | FRED `CPIAUCSL` | Monthly |
| Daily S&P 500 returns | yfinance, daily interval | Used to build `svar` |

**Derived variables you must compute, not fetch directly:**
- `tms` (term spread) = long-term yield − T-bill rate
- `dfy` (default yield spread) = BAA yield − AAA yield
- `infl` (inflation) = percent change in CPI, **shifted forward by one month** (see "Common Mistakes" below — this one is a classic leakage trap straight from the paper itself)
- `svar` (stock variance) = sum of squared **daily** S&P 500 returns within each month (this requires fetching daily data separately from your monthly data, then resampling)

### Design decisions to make yourself
1. What should the function signature look like? Think about what parameters a caller (your future API) will need: start date, end date, maybe which predictors to include.
2. How will you align all these different series, which come from different sources with possibly different missing dates? (Hint: an inner join on the date index, after resampling everything to the same monthly frequency, is the cleanest approach.)
3. What format should monthly dates be in so joins work cleanly? (Hint: normalize every index to the first day of the month before joining — mismatched day-of-month is a classic silent join failure.)

### Definition of Done
Calling your fetch function with a start and end date returns a single DataFrame, indexed by month, with no unexpected NaNs in the middle of the series (some NaNs at the very start are expected and fine — that's just data availability).

### Common Mistakes
- **The inflation lag.** Welch & Goyal explicitly note that CPI inflation data is released with a one-month delay — you don't know October's inflation until sometime in November. If you don't shift this variable forward by one period, you're letting the model "know" the inflation number before it would have actually been published. This is real-world look-ahead bias hiding in a paper-following detail.
- **Resampling daily data with the wrong aggregation.** When resampling daily S&P returns to monthly for `svar`, you want the *sum of squared* returns, not the mean or the last value. Using `.resample("MS").last()` here would be wrong — you'd just get one day's squared return instead of the whole month's variance.
- **Forgetting `auto_adjust=True`** (or whatever the equivalent is in your yfinance version) when pulling prices — you want dividend- and split-adjusted prices, otherwise your return calculations will have artificial jumps around split dates.
- **Joining before resampling.** If you join a monthly series with a daily series directly, you'll get a mess of NaNs or duplicated rows. Always resample to a common frequency *before* joining.

---

## Step 2 — Welch & Goyal OOS R² Benchmark

### Goal
For a single macro predictor variable, compute whether using it in a simple regression actually beats just predicting "the historical average" — out of sample.

### What you need conceptually

This is the mathematical heart of Paper 1. Walk through the logic slowly:

**The competing models:**
1. **Naive model**: at any point in time, predict next period's equity premium as the *average of all premiums observed so far* (an "expanding window" mean — it grows as you add more historical months).
2. **Conditional model**: fit a simple linear regression of equity premium on the lagged predictor, using only data available up to that point, then use it to predict the next period.

**The critical lag:** the predictor used to forecast time T's premium must be the predictor value known at time T−1. If you use the value at time T, you're using information that wasn't available yet when the forecast would have been made.

**The expanding window procedure** (this is the part people get wrong):
- Start with some minimum amount of training data (the paper and most replications use a handful of years before producing the first forecast).
- At each subsequent time step: fit a model using *all data up to and not including the current step*, predict the current step, record the prediction and the actual outcome, then move forward one step and refit using one more observation.
- This means you are refitting the regression at every single time step. It's not one model fit once — it's potentially hundreds of models, one per OOS observation.

**The R² formula itself:**
```
OOS R² = 1 − (MSE of conditional model) / (MSE of naive model)
```
Where MSE is computed only over the out-of-sample period (never over the training data used to fit each model).

**Interpreting the result:**
- Positive → the predictor adds real value over just using the historical average
- Zero → no better than the naive guess
- Negative → actively worse than the naive guess (this is what WG found for most predictors)

### Design decisions to make yourself
1. How will you structure the loop that refits the model at every time step? Think about what's "training data" at each iteration — it's not a fixed window, it grows by one observation each time.
2. What's your minimum training window before you start producing OOS predictions? Too short and your early regressions are unstable; too long and you waste your already-limited sample.
3. How will you store results so you can later plot the *cumulative* R² over time (not just the single final number)? You'll need to track cumulative squared errors for both models as you go.
4. How should this work across multiple predictors? Think about writing one function that handles a single predictor cleanly, then a second function that loops over all your available predictors and assembles a summary table.

### Definition of Done
You can pass in any single predictor column name and get back: a single OOS R² number, and a time series you could plot showing how that R² evolved as more out-of-sample months accumulated. Running this across all six-ish macro predictors gives you a small leaderboard table, and most of them should come out negative or barely positive — if everything is wildly positive, be suspicious (see below).

### Common Mistakes
- **Forgetting to lag the predictor.** This is the single most common bug. Double check: at the iteration where you're predicting month T, the X value you feed into `.predict()` must be the predictor's value from month T−1, not month T.
- **Fitting the naive model on the full sample instead of expanding it.** The naive "historical average" must also only use data up to the current point in time — it can't peek at the future either. A model that's allowed to know the *full-sample* average is itself leaking information.
- **Suspiciously high R².** If you get an OOS R² above a few percent for a monthly macro predictor, you very likely have a leakage bug (probably the lag issue above). Real-world monthly equity premium R² values are small — often slightly negative, occasionally a percent or two positive in good cases.
- **Refitting the model only once instead of every period.** It's tempting (and much faster) to fit one regression on the whole training set and use it for every OOS prediction. That's not what the paper does and not what "expanding window" out-of-sample evaluation means — it changes the entire interpretation of the test.

---

## Step 3 — Stock-Level Feature Engineering (GKX-style)

### Goal
Take raw daily price/volume data for a single ticker and produce a rich feature table where, for any given date, every single feature value could have been computed by someone standing on that date with no knowledge of the future.

### What you need conceptually

Group your features into the same families GKX use. For each, think through *why* it might predict returns, not just how to calculate it.

**Momentum / reversal features**
- Short-term reversal (last ~1 month return): stocks that just dropped hard sometimes bounce, and vice versa — driven partly by liquidity effects, not necessarily fundamentals.
- Medium-term momentum (e.g., trailing ~12 months, but explicitly excluding the most recent month): stocks that have been rising tend to keep rising over months-long horizons. The reason you exclude the most recent month is that combining short-term reversal and long-term momentum in the same window would muddy two genuinely different effects.
- Think about why "excluding the most recent month" requires you to shift your lookback window, not just shorten it.

**Volatility features**
- Realized volatility over different windows (e.g., ~1 month, ~3 months): the standard deviation of daily returns, annualized.
- Idiosyncratic volatility: the volatility of a stock's returns *after* removing the part explained by the overall market's movements (i.e., the volatility of the regression residual from regressing the stock's returns on market returns). This requires you to first estimate the stock's beta.
- Market beta: the sensitivity of the stock to market moves, typically estimated via a rolling regression. Think about what window length makes sense (too short = noisy estimate, too long = stale estimate that doesn't reflect a changing company).

**Liquidity features**
- Dollar volume: price times shares traded — a simple proxy for how easy it is to trade a stock without moving its price.
- Illiquidity measures (e.g., the relationship between absolute return and trading volume): bigger price moves per dollar traded implies a less liquid stock.

**Price-level / technical features**
- Distance from the 52-week high.
- Moving average relationships (e.g., is the price above or below its 50-day average, and how does the 50-day average compare to the 200-day average).
- An oscillator like RSI, which measures whether recent price action has been disproportionately up or down moves.

**Macro interaction features**
- GKX's key insight that you should replicate conceptually: a feature's predictive power may *depend* on the macro environment. For example, momentum might work differently when interest rates are rising versus falling. Practically, this means multiplying a stock-level feature by a macro variable (like the term spread you built in Step 1) to create an interaction term.
- This requires you to bring your monthly macro data into a daily-frequency stock dataset. Think carefully about which direction to fill missing values — you only learn this month's macro reading once it's published, so within a month you should be carrying forward the most recent *known* value, never reaching forward into a future reading.

**Normalization step**
- GKX cross-sectionally rank-normalize every feature (across thousands of stocks at each point in time) into a fixed range. Since you're working with one stock at a time, you'll need an analogous idea: convert each raw feature into a percentile *relative to its own history up to that date* — using an expanding window, not the full-sample distribution (which would again leak future information into past feature values).

### Design decisions to make yourself
1. What's the natural unit for each "lookback window"? Decide whether you're thinking in trading days or calendar days, and be consistent — pandas rolling functions on daily data usually want a number of rows (trading days), not calendar days.
2. How will you handle the beta/idiosyncratic volatility calculation, which requires *external* market data (you'll need to separately fetch a market proxy like SPY)? Think about how to align two differently-shaped DataFrames (your stock and the market index) by date before doing the rolling covariance/variance calculation.
3. How will you prevent a feature from "looking into the future" by even one row? A good personal discipline: for every single feature, explicitly ask "if I were standing on this exact date, would I have access to this information yet?"
4. Outliers: will you clip extreme values? GKX do this — think about why unbounded features (a stock that doubles in a day) might destabilize a downstream linear or even tree-based model.

### Definition of Done
You have a daily-indexed feature table with ~15-25 columns. Spot check at least three feature columns by hand on a single date: pick a date, manually verify the rolling window only includes days strictly before (or at) that date, and confirm the value makes intuitive sense (e.g., if the stock has been steadily rising, your momentum feature should be positive).

### Common Mistakes
- **Off-by-one errors in shifts.** This family of features is where shift/lag mistakes are easiest to make invisibly, because the resulting numbers still "look reasonable" — they're just one day ahead of where they should be. The only reliable way to catch this is manual spot-checking specific dates.
- **Computing momentum without skipping the most recent month.** If you compute "trailing 12-month return" as a single rolling window ending today, you've accidentally combined the reversal effect and the momentum effect into one number, which is not what the literature intends and which dilutes the signal.
- **Using `.fillna()` or interpolation across the macro-merge in a way that fills *backward*.** Always forward-fill macro data into daily frequency, never backward-fill — backward-filling means a daily row in early March could pick up a macro reading that wasn't published until late March.
- **Rank-normalizing using the full sample.** If you compute each feature's percentile rank using the entire dataset (including future dates) instead of an expanding window up to the current date, you've leaked future information through the back door, even though the underlying raw feature was computed correctly.
- **Forgetting that the market proxy (SPY) needs its own return series, not its price series, when computing beta.** Beta is a relationship between *returns*, not raw price levels.

---

## Step 4 — Defining the Prediction Target

### Goal
Create the variable you're actually trying to predict, and join it to your features in a way that has zero overlap between "what the model can see" and "what it's trying to guess."

### What you need conceptually

The target is a **forward-looking return** — e.g., "what will the return be over the next N trading days, starting from today." This is fundamentally different from every feature you built in Step 3, which all look *backward* from today.

The mechanical pattern: you compute a return that requires data N days into the future, then shift it backward so that it's stored on *today's* row, representing "this is what happens starting from today."

After joining your features and target onto a single table, indexed by date, the last N rows of your dataset will have a missing target (because you don't have N days of future data after the last day in your sample yet) — and these rows must be dropped, not filled with anything.

### Design decisions to make yourself
1. What prediction horizon will you support first? Building this to work for one horizon (e.g., ~1 month) before generalizing to multiple horizons will save you debugging headaches.
2. How will you structure the function so that a future caller (your API) can pass in a horizon and get back a clean modeling-ready dataset?

### Definition of Done
You can produce a single table where every row has: a date index, ~15-25 feature columns (all using only past information as of that date), and one target column (the future return). When you print the *last few rows* of this table, the target column should be NaN — and you should explicitly drop these before modeling, with a comment to yourself explaining why.

### Common Mistakes
- **Joining features and target by row position instead of by date index.** This is "silently disastrous" — if your feature table and your target series get misaligned by even one row, every single training example becomes corrupted, and your model will appear to "work" (because it's accidentally learning the right answer is one row away) until you deploy it and it falls apart.
- **Not dropping the trailing rows with missing targets**, and instead filling them with zero or the mean. This corrupts your test set with fake "the future was flat" examples.

---

## Step 5 — Temporal Train/Validation/Test Split

### Goal
Split your modeling dataset into three chronological chunks, with absolutely no shuffling.

### What you need conceptually

This is the single most important habit to build correctly before you train any model, because every other mistake in this project is recoverable, but a broken split silently invalidates every result downstream.

**Why not the usual `train_test_split` with `shuffle=True` that you're used to from generic ML coursework?** In most ML problems, each row is independent (an image, a customer record). In time series, rows are *not* independent — row T and row T+1 are highly correlated, because your features use overlapping rolling windows. If you shuffle, your training set may end up containing a date that's only three days away from a date in your test set. Even though they're technically different rows, the features practically leak across the split because of all that rolling-window overlap. Your model will appear to perform far better than it actually would in real deployment.

**The structure you want:** picture your full date range as a single timeline. Training data is the earliest chunk, validation is the middle chunk, test is the most recent chunk. No overlap, no shuffling, strict chronological order.

**What each split is for:**
- Training: where model parameters (e.g., regression coefficients, tree splits) are actually fit.
- Validation: where you choose hyperparameters (e.g., how much regularization, how many trees) by checking which choice performs best on data the model hasn't been fit on — but you're still allowed to look at the validation result many times as you experiment.
- Test: touched exactly once, at the very end, to report the final number. If you go back and re-tune anything after looking at the test set performance, the test set is no longer "out of sample" in any meaningful sense — you've effectively turned it into a second validation set.

### Design decisions to make yourself
1. What proportions will you use for each split? Think about your total sample size — with only a few years of daily data for one ticker, you may have fewer usable rows than you'd like, so consider this when deciding how much you can afford to reserve for validation and test.
2. How will any preprocessing step that "learns" something from data (most notably, feature standardization/scaling) interact with this split? Think hard about this one before moving on — it's covered in detail below in Step 6.

### Definition of Done
You have a function or clear code path that, given your full modeling dataset, returns three separate DataFrames where the maximum date in the training set is strictly before the minimum date in the validation set, which is strictly before the minimum date in the test set. Print out the date ranges of all three and visually confirm there's no overlap.

### Common Mistakes
- **Using `random_state` based splitting at all**, out of habit from other ML coursework. There should be no randomness in *which rows* go into which split — only in things like neural network weight initialization.
- **Allowing any gap-free assumption to break.** If your raw data has gaps (e.g., missing days), make sure the split is based on the actual dates present, not on row *counts* under an assumption of continuity.

---

## Step 6 — Preprocessing Without Leakage (Scaling)

### Goal
Standardize/scale your features in a way that respects the train/val/test boundary.

### What you need conceptually

Many models (Ridge regression and neural networks especially) perform much better when features are on comparable scales — otherwise a feature like "dollar volume in the billions" can dominate a feature like "RSI between 0 and 100" purely due to magnitude, not actual predictive importance.

**The rule:** any transformation that "learns" statistics from your data (like a mean and standard deviation for standardization) must learn those statistics *only* from the training set. You then apply that *same* learned transformation to the validation and test sets, without ever recomputing it on them.

**Why this matters so much:** if you instead compute the mean/std using your *entire* dataset (train+val+test combined) before splitting, you've let information about the future (the test set's statistical properties) subtly influence how your training data gets transformed. This is a quieter form of leakage than the others, because the model still "sees" data point values, not the test set's actual outcomes — but it's leakage all the same, and it can meaningfully inflate your reported performance.

### Design decisions to make yourself
1. At what point in your pipeline does scaling happen — before or after the train/val/test split? (It should always be after the split, fit only on train.)
2. If you later want to retrain using train+validation combined (a common practice once you've locked in your hyperparameters, before final test evaluation), how does that affect when you refit your scaler?

### Definition of Done
You can clearly articulate, for any preprocessing step in your pipeline, exactly which subset of data it was fit on. If the answer to "was this fit on anything from validation or test" is ever yes, stop and fix it before training your first model.

### Common Mistakes
- **Calling the "fit" version of your scaler on validation or test data.** Many scaling libraries have both a "fit" method (learns statistics) and a "transform" method (applies already-learned statistics) — make sure you only ever call "fit" on the training set, and "transform only" everywhere else.
- **Recomputing normalization features (like the expanding-window rank normalization from Step 3) without realizing they have their own leakage risk** if not built carefully — revisit Step 3's "Common Mistakes" if this feels fuzzy.

---

## Step 7 — Model 1: Ridge Regression

### Goal
Train a regularized linear model as your simplest, most interpretable baseline ML model.

### What you need conceptually

**Why not plain linear regression (OLS)?** With a feature set of 15-25 (possibly correlated) predictors and a modest number of observations, plain OLS tends to overfit — it will find some combination of coefficients that fits the training noise extremely well, but that combination won't generalize. This mirrors exactly what GKX found at a much larger scale: OLS performance collapsed once they expanded to hundreds of predictors.

**What Ridge does differently:** it adds a penalty to the regression's objective that punishes large coefficient values. The practical effect is that coefficients get pulled toward (but not all the way to) zero, which trades off a small amount of training fit for a large amount of stability — meaning the fitted model is less sensitive to noise in any particular training sample, and tends to generalize better.

**The key hyperparameter** is the strength of that penalty. Think about the two extremes: a penalty of (near) zero recovers plain OLS (overfitting risk returns); an enormous penalty crushes every coefficient toward zero (the model just predicts something close to a constant, ignoring the features — underfitting). The right value is somewhere in between, and you find it empirically.

### Design decisions to make yourself
1. How will you search over candidate penalty strengths? Think about a reasonable range — penalty strengths in these kinds of problems often need to span several orders of magnitude (e.g., very small to very large) since you don't know in advance where the right value lives.
2. How will you select the "winning" penalty strength? (Hint: evaluate each candidate's performance on the validation set, not the training set — selecting based on training performance just brings back the overfitting problem you were trying to avoid.)
3. Once you've picked the best hyperparameter, should you refit the final model using only the training data, or training+validation combined? Think about why using more data for the final fit (now that you're done tuning) is generally beneficial, as long as you're disciplined about which data informed which decision.

### Definition of Done
Given your train/val data, you can report a single chosen penalty strength along with the validation R² it achieved at that hyperparameter, and you have a "final" fitted model object ready to evaluate on the untouched test set later.

### Common Mistakes
- **Choosing the penalty strength based on test-set performance.** This is one of the most common subtle leaks in ML projects generally, not just finance —"hyperparameter tuning on the test set" defeats the entire purpose of having a test set.
- **Forgetting to scale features before fitting Ridge.** Ridge's penalty treats all coefficients somewhat equally in magnitude terms, so if your features are on wildly different scales, the penalty will affect them very unevenly, distorting which features get "selected."

---

## Step 8 — Model 2: Random Forest

### Goal
Train a tree-based ensemble model capable of capturing nonlinear relationships and interactions that Ridge cannot.

### What you need conceptually

**Why might a Random Forest do better here?** GKX's central empirical finding is that *nonlinear interactions* between predictors matter a lot for return prediction — for example, momentum might only "work" when volatility is low. A linear model like Ridge can never represent this kind of "it depends on a combination of two other features" relationship; tree-based models can.

**How a Random Forest works, conceptually:** it builds many individual decision trees, each trained on a slightly different random subset of the data and features, then averages their predictions. The randomness across trees is what makes the *ensemble* more stable than any single tree (which would badly overfit on its own).

**Key hyperparameters to think through:**
- Number of trees: generally, more is better up to a point of diminishing (and eventually negligible) returns, at the cost of speed.
- Tree depth / minimum samples per leaf: this controls how complex each individual tree is allowed to get. GKX specifically found that in financial prediction problems (low signal-to-noise), shallow trees with very few leaves perform best — deep trees just memorize noise. This is an important and slightly counterintuitive lesson: more model complexity is not automatically better when the underlying signal is faint.
- Fraction of features considered at each split: this controls how different the individual trees are from each other, which affects how much the ensemble benefits from averaging.

### Design decisions to make yourself
1. Following the same logic as Ridge: how will you search the hyperparameter space, and on what data will you evaluate each candidate combination?
2. Given GKX's finding about shallow trees mattering in this domain, what range of depth values should you actually bother searching? (Hint: don't bother including very deep trees in your search grid — you already have a strong prior reason to expect they'll overfit on a small, noisy dataset.)
3. How will you extract and interpret feature importances from the final model? Think about what "importance" actually measures for a tree ensemble (roughly: how much each feature contributed to reducing prediction error across all the trees and splits) — and be ready to sanity check whether the most "important" features make intuitive sense (e.g., if momentum features dominate, that's consistent with the literature; if some oddball feature dominates, double check it for a leakage bug).

### Definition of Done
You have a fitted Random Forest with hyperparameters chosen via validation performance, plus a ranked list of feature importances you can eyeball for sanity.

### Common Mistakes
- **Letting trees grow unrestricted ("max depth = None").** This is the default in many libraries and is almost always wrong for this kind of problem — it will severely overfit on a small, noisy financial dataset.
- **Treating high feature importance as proof of real predictive power without skepticism.** A feature can get high importance because it happens to correlate with noise in your specific training window. This is exactly why the validation and test set evaluations exist — importance scores are a diagnostic, not a final verdict.

---

## Step 9 — Model 3: Neural Network

### Goal
Train a small feedforward network as your most flexible model, while respecting the lesson from GKX that bigger is not better here.

### What you need conceptually

**Why a neural network at all, given you already have Random Forest for nonlinearity?** Neural networks can represent a different and sometimes complementary class of nonlinear functions, and GKX found neural nets among the top performers (alongside trees) in their large-scale study. For your project, it's also valuable pedagogically — wiring up a clean PyTorch training loop with proper validation-based early stopping is a core skill in itself.

**The architecture lesson from the paper:** GKX tested networks from one to five hidden layers and found performance peaked around three hidden layers, then *declined* with more depth. This is the opposite of what you might expect from typical deep learning intuition (where "more layers, more data, better performance" is often the story) — but financial return data has an extremely low signal-to-noise ratio compared to, say, image data, so extra model capacity mostly just means more capacity to memorize noise.

**Critical training concept — early stopping:** unlike Ridge or Random Forest, a neural network is trained iteratively over many passes through the data ("epochs"). If you let it train for too many epochs, it will eventually start memorizing training-set noise even if it generalized well at an earlier point in training. The fix: after every epoch (or every few epochs), check performance on the validation set, and keep track of the *best* validation performance seen so far along with a snapshot of the model at that point. If validation performance stops improving for a while, stop training early and roll back to that best snapshot — don't just use whatever the model looks like after the last epoch.

**Other ingredients worth understanding, not just including:**
- Batch normalization / dropout: regularization techniques that help prevent overfitting, conceptually analogous in spirit to Ridge's penalty term, but achieved through different mechanisms.
- Weight decay (often built into the optimizer): another form of the same "penalize large weights" idea from Ridge, applied inside a neural network.
- Gradient clipping: a stability technique that prevents any single noisy batch of data from causing a wildly oversized parameter update.

### Design decisions to make yourself
1. How many hidden layers and how wide should each be? Given the paper's finding, start small (e.g., two or three modest-sized hidden layers) rather than reaching for something large by default.
2. How will you implement the "track best validation performance, stop if it hasn't improved in a while" logic? Think through the bookkeeping: you need to store the best score seen so far, the model weights at that point, and a counter for how many epochs have passed without improvement.
3. What will you do differently for the neural network's scaling/preprocessing step compared to Ridge and Random Forest? (Hint: scaling matters even more for neural networks than for Ridge — think about why unscaled inputs are especially harmful to gradient-based optimization.)

### Definition of Done
Your network trains without diverging (loss should generally trend down then plateau, not explode to huge numbers or NaN), your early-stopping logic correctly identifies and restores the best-validation-performance snapshot, and you can report the validation R² of that snapshot model the same way you did for Ridge and Random Forest.

### Common Mistakes
- **Reaching for a deep, wide network "to be safe."** Re-read the architecture lesson above. In this specific domain, that instinct is usually counterproductive.
- **Using the final-epoch model instead of the best-validation-epoch model.** Without explicit early-stopping logic, your "final" model is just whatever happened to be the weights after your last training pass — likely already past the point of overfitting.
- **Forgetting `model.eval()` mode (or your framework's equivalent) when computing validation/test predictions**, if you're using batch normalization or dropout. These layers behave differently during training versus evaluation, and forgetting to switch modes will silently corrupt your validation metric. Always treat batch norm / dropout layers as a place to double-check this for your model.

---

## Step 10 — Final OOS Evaluation and Model Comparison

### Goal
Evaluate all three trained models — Ridge, Random Forest, Neural Network — on the test set, exactly once, and compare them on equal footing.

### What you need conceptually

This is where you find out which model "wins," using the exact same out-of-sample R² logic from Step 2, just applied to your stock-level ML models instead of WG's macro regressions.

**The fairness requirement:** all three models must be evaluated on the *identical* test set rows, using the *identical* target definition, so that differences in R² reflect genuine differences in model quality rather than differences in what each model was tested against.

**What "winning" should mean to you:** don't just look at which model has the highest test R². Also sanity check: is the winning model's test R² in a believable range (small but positive, similar order of magnitude to what the paper reports), or absurdly high (likely a leakage bug somewhere upstream)? A model that wins by being "too good" should make you suspicious, not happy.

### Design decisions to make yourself
1. How will you structure a single evaluation function that works across all three model types, given that a scikit-learn model and a PyTorch model have different prediction interfaces? Think about writing a thin wrapper or adapter rather than copy-pasting near-identical evaluation code three times.
2. How will you present the comparison? At minimum you want, side by side: each model's test R², and ideally each model's validation R² alongside it (to check whether validation and test performance broadly agree — if a model did great on validation but terribly on test, that's worth investigating rather than ignoring).

### Definition of Done
You have one small table (even just printed to console) showing Ridge, Random Forest, and Neural Network test-set OOS R² values side by side, computed on the same test rows, each using the model's properly-fitted (and properly-scaled, where relevant) artifacts from earlier steps.

### Common Mistakes
- **Touching the test set more than once during model development.** If you evaluate on test, don't like the result, go back and change a hyperparameter, then re-evaluate on test again — you've turned your test set into a second validation set, and your final reported number is now optimistic. The discipline here is mostly about workflow, not code: decide your hyperparameters using only validation, and treat the test evaluation as something you run once near the end.
- **Comparing models evaluated on different row counts** (e.g., because one model's preprocessing dropped a few additional rows due to NaNs). Double check all three models are scored against an identical set of test dates.

---

## Step 11 — Simple Backtest: Strategy vs. Buy-and-Hold

### Goal
Convert your model's return predictions into a simple trading strategy, and compare its cumulative performance against just holding the stock the entire time.

### What you need conceptually

**The simplest possible strategy logic:** on any day where your model predicts a positive future return, take a "long" position (matching how a buy-and-hold investor is positioned); on any day where it predicts a negative future return, hold cash instead (or, slightly more advanced, go short — but start with the cash version, it's simpler to reason about and debug).

**Why this is a meaningful test beyond R²:** a model can have a positive but tiny R² and still produce a strategy that meaningfully beats buy-and-hold, if it's especially good at avoiding the worst periods (even without being especially good at predicting the best periods). Conversely, a model with a "good-looking" R² could still produce a mediocre strategy if its sign predictions (up versus down) aren't reliable, even if its magnitude predictions are decent. Building both the OOS R² check (Step 10) and the backtest gives you two different, complementary views of the same model.

**Computing cumulative performance:** at each time step in your test period, your strategy earns either the actual realized return (if it was "long" that period) or roughly nothing (if it was in cash that period). The buy-and-hold benchmark always earns the actual realized return. To compare them visually, convert each day's return into a *running cumulative product* (starting from a value of 1.0) for both the strategy and the benchmark — this turns a series of period returns into a single growth curve you can plot.

**A risk-adjusted comparison metric (Sharpe ratio):** raw cumulative return doesn't account for how bumpy the ride was to get there. The Sharpe ratio — average return divided by the standard deviation of returns, appropriately annualized — lets you compare whether the strategy's returns came with proportionally more or less risk than buy-and-hold's. Think about why a strategy with a slightly lower total return but much smoother ride might still be the "better" strategy by this measure.

### Design decisions to make yourself
1. Should your strategy use the model's *predicted sign* only (long vs. cash), or scale position size by the predicted *magnitude* of the return? Starting with sign-only is simpler and a perfectly reasonable first version — note for yourself that magnitude-scaling is a natural improvement to revisit later.
2. Over what time period are you computing this backtest — should it be restricted to your test set only? (Yes — backtesting over data the model was trained on would just be re-demonstrating in-sample fit, not genuine strategy performance.)
3. How will you handle the annualization factor in your Sharpe ratio calculation, given that your return horizon (e.g., ~1 month) isn't daily? Think through how many non-overlapping periods of that horizon length fit into a year, and use that to scale your ratio appropriately for an apples-to-apples annualized comparison.

### Definition of Done
You can produce two cumulative-return curves — your strategy and buy-and-hold — covering only the test period, plus a Sharpe ratio for each. Eyeball the chart: does the strategy curve diverge from buy-and-hold in a way that's consistent with the sign of your model's test R² (i.e., if your model's R² was barely positive, you shouldn't expect a dramatically superior backtest — be suspicious if you see one)?

### Common Mistakes
- **Backtesting over the training period (even by accident).** Always double, triple check the date range your backtest is iterating over matches your test set exactly.
- **An unrealistically perfect-looking backtest.** If your strategy curve looks dramatically, suspiciously better than buy-and-hold while your underlying model R² was small, retrace your steps — there's likely a subtle leakage bug feeding slightly-future information into either your features or your strategy logic (a very common one: accidentally using today's return to decide today's position, instead of yesterday's prediction to decide today's position).

---

## Step 12 — Monte Carlo Simulation (GBM)

### Goal
Simulate thousands of possible future price paths for a stock, using your ML model's return and volatility estimates as the simulation's parameters, and summarize the result as percentile bands.

### What you need conceptually

**The underlying model: Geometric Brownian Motion (GBM).** This is a standard way to simulate stock prices that assumes returns are driven by two ingredients: a steady "drift" (the expected return) and a random "shock" each period (driven by volatility). Each simulated path takes today's price and repeatedly multiplies it by a small random daily growth factor, day after day, to project a path forward.

**The math you need to understand, not just implement — walking through it step by step:**

1. You start with an annualized expected return (call it `mu`) and an annualized volatility (call it `sigma`) — these come from your ML model's outputs, not from a generic historical average, which is the specific twist this assignment wants beyond a textbook GBM implementation.

2. You need to convert these annual figures into a *daily* drift and a *daily* volatility, because you're simulating day-by-day. Volatility scales with the square root of time, while drift scales linearly with time — think through why these two quantities have different time-scaling relationships (it comes from how variance vs. standard deviation each behave under the assumption that daily returns are roughly independent of each other).

3. There's a subtlety in the daily drift calculation that's easy to skip but mathematically important: the *naive* approach of just dividing the annual return by the number of trading days in a year is not quite right once you're working in log-return space (which GBM does, because multiplying many small growth factors together is mathematically equivalent to adding their logarithms). You need to subtract a small correction term related to half of the variance. Try to understand *why* this correction exists — it relates to the difference between the average of a set of returns and the return that corresponds to the average of the *logs* of those returns (these are not the same number, and the gap between them grows with volatility). If you skip this correction, your simulated median price path will systematically drift away from what your `mu` parameter implies.

4. For each one of your (e.g., 10,000) simulated paths, and for each day within your chosen time horizon, you draw a random shock from a standard normal distribution, combine it with your daily drift and volatility, and that gives you that day's simulated log-return. Summing up all the day's log-returns from day zero up to any given day, then exponentiating, tells you the cumulative growth factor to apply to today's actual price, to get a simulated price on that future day.

5. Once you have a large grid of simulated prices (one row per simulated path, one column per simulated day), you can compute, for each future day independently, percentiles across all the simulated paths (e.g., the 10th percentile price, the 50th/median price, the 90th percentile price on that day). Plotting these percentile values across all days produces the "fan chart" shape — narrow near today (since all paths start at the same known price) and progressively widening further into the future (since uncertainty compounds over time).

### Design decisions to make yourself
1. How will you vectorize the simulation so that generating 10,000 paths across hundreds of days doesn't require slow nested Python loops? (Hint: think about generating *all* the random shocks for *all* paths and *all* days in one large array operation up front, rather than looping day-by-day and path-by-path.)
2. What percentiles will you report, and how will the user adjust the time horizon? Think about how your function's interface needs to expose the time horizon as an adjustable parameter, since the project requires a 30/90/180/365-day slider in the eventual UI.
3. Where will `mu` and `sigma` come from in your actual pipeline? The assignment specifically wants these parameterized by your *ML model's* output, not a simple historical average — think about how you'll plumb your Step 10 model's predicted return and your Step 3 volatility feature through to this simulation function, versus using them only as a fallback if the ML connection isn't ready yet.

### Definition of Done
Given a starting price, an annualized return, an annualized volatility, and a number of days, you can produce a grid of simulated price paths and summarize them into at least a 10th/50th/90th percentile band per day. Plot it (even crudely) and visually confirm the classic "fan" shape: narrow at day zero, wide by the final day.

### Common Mistakes
- **Forgetting the variance correction term in the drift calculation**, leading to a median simulated path that doesn't match your intended `mu`.
- **Mixing up annualized and daily parameters somewhere in the pipeline** — a very easy mistake when juggling both scales. Adding a clear comment or variable naming convention (e.g., explicitly naming variables `annual_mu` vs `daily_drift`) will save you real debugging time here.
- **Looping over days and simulations with nested Python for-loops.** This will work but will be painfully slow at 10,000 simulations — push yourself to find the vectorized approach instead, both for performance and because it's a useful numpy skill in itself.

---

## Step 13 — DCF Valuation Model

### Goal
Estimate a company's "intrinsic value per share" using a simplified discounted cash flow model, driven by a small number of user-adjustable assumptions.

### What you need conceptually

**The core idea of DCF, explained from first principles:** a business is worth the sum of all the cash it will ever generate for its owners, adjusted for the fact that a dollar received further in the future is worth less than a dollar received today. This adjustment is called "discounting," and the rate used to discount is the WACC (Weighted Average Cost of Capital) — conceptually, the minimum annual return investors would require to be willing to fund this business, blending the cost of its debt and the cost of its equity.

**Why discount at all? A simple intuition:** if you could put money in a safe investment earning, say, 8% a year, you wouldn't value a promise of $108 one year from now as being worth $108 today — you'd only pay about $100 today, since that $100 could grow into $108 on its own. The further out the cash flow, and the higher the required return, the smaller today's "fair price" for that future cash flow becomes. This is exactly what dividing by `(1 + WACC)` raised to a power (the number of years out) accomplishes mathematically.

**The structure of a simplified 5-year DCF:**
1. **Project future free cash flow (FCF) for five years**, starting from the company's current FCF and growing it at an assumed growth rate each year. (Free cash flow, roughly: cash generated from normal operations, minus the cash spent on maintaining/growing the business's physical assets.)
2. **Discount each of those five years' projected FCF back to today's dollars**, using the WACC.
3. **Estimate a "terminal value"** representing everything that happens *beyond* year five, since a business doesn't simply stop existing after your explicit forecast window. The standard simplified approach (the Gordon Growth Model) assumes that, beyond year five, cash flow grows at some smaller, more sustainable long-run rate forever, and there's a known formula for converting an infinitely growing stream of cash flow into a single present-day number.
4. **Discount that terminal value back to today as well** (since it represents a value *as of* year five, not today).
5. **Add everything up** — discounted years 1-5 plus discounted terminal value — to get the total estimated value of the underlying business (enterprise value).
6. **Adjust for debt and cash** to convert "value of the business" into "value belonging to shareholders" (equity value) — subtract debt (since debt holders have a claim ahead of shareholders) and add back cash (since cash on the balance sheet belongs to shareholders).
7. **Divide by shares outstanding** to get a per-share intrinsic value you can compare against the current market price.

**The Gordon Growth terminal value formula, conceptually:** the present value (as of year 5) of an infinitely growing perpetual cash flow stream depends on next year's expected cash flow, divided by the gap between your discount rate and your assumed perpetual growth rate. Think carefully about why this formula *requires* your discount rate to be strictly greater than your perpetual growth rate — what happens mathematically (and intuitively, in terms of "a business growing faster than the economy forever") if growth equals or exceeds the discount rate?

**Margin of safety:** once you have an intrinsic value per share, compare it to the actual current market price. If intrinsic value is meaningfully above the market price, the stock might be undervalued by this model's assumptions (and vice versa). Express this gap as a percentage for an intuitive headline number.

### Design decisions to make yourself
1. What real financial data will you need to pull for any given ticker (current free cash flow, shares outstanding, existing debt and cash levels), and what's a reasonable plan if any of these are missing or look unreliable for a given company (which happens often with free data sources)?
2. How will you structure your function's inputs so that WACC, the 5-year growth rate, and the terminal growth rate are all easily adjustable — since the project explicitly wants these as user-controlled sliders eventually?
3. For the waterfall chart described in the requirements, what are the natural "buckets" to break the final valuation into? Think about which intermediate quantities in your calculation (each year's discounted cash flow contribution, the discounted terminal value, the debt/cash adjustment) would make a meaningful, intuitive breakdown for a chart showing "how did we get to this final number."

### Definition of Done
Given a ticker's basic financials and a set of assumptions (WACC, 5-year growth, terminal growth), you can compute and clearly report: enterprise value, equity value, intrinsic value per share, and a margin of safety percentage versus the current market price. You can also produce the handful of intermediate numbers needed to break the total value into the "Year 1-5 cash flows" piece, the "terminal value" piece, and the "debt/cash adjustment" piece.

### Common Mistakes
- **Letting WACC be less than or equal to the terminal growth rate.** As discussed above, the Gordon Growth formula breaks down (produces a negative or infinite "value") in this case — make sure your code explicitly guards against this and communicates a clear error rather than returning a nonsensical number silently.
- **Treating free, automatically-pulled financial data as gospel.** Real-world financial statement data, especially free cash flow figures from automated sources, can be noisy, restated, or simply wrong for some companies. Make sure your eventual webapp is explicit with the user that this is a simplified, assumption-driven model — not investment advice — exactly as the project requirements specify.
- **Forgetting the debt/cash adjustment entirely**, and dividing enterprise value straight by shares outstanding. This skips an important step — enterprise value represents the whole business (including the portion effectively "owned" by debt holders), not just the equity portion that belongs to shareholders.

---

## Step 14 — FastAPI Backend

### Goal
Wrap everything you've built (Steps 1-13) behind a small set of HTTP endpoints that a frontend can call.

### What you need conceptually

**Think of this step as "plumbing," not new logic.** You're not inventing new financial or ML concepts here — you're exposing the functions you already wrote behind a clean interface so a separate frontend application (which won't share Python memory with your backend) can request results.

**What each endpoint roughly needs to do:**
- Accept some input from the caller (at minimum: a ticker symbol, and likely a date range and/or a prediction horizon).
- Call into your Step 1-13 functions, in the right order, with those inputs.
- Package the results into a format that's easy for a frontend to consume — think about what shape of data a chart-plotting library typically wants (e.g., a list of dates alongside a parallel list of values, rather than, say, a raw pandas DataFrame, which a frontend can't directly use).
- Handle the case where something goes wrong (bad ticker, insufficient data, a network failure pulling external data) gracefully, returning a clear error rather than crashing silently.

**You'll likely want (at least) one endpoint per tab:** one for the Welch & Goyal baseline table and curves, one for the ML model comparison and backtest, one for the Monte Carlo simulation, and one for the DCF valuation.

### Design decisions to make yourself
1. What should the *request* shape look like for each endpoint — i.e., what does the frontend need to send you? Sketch this out before writing any endpoint code; it'll also clarify what your frontend needs to collect from the user.
2. What should the *response* shape look like? Specifically think about how you'll serialize date-indexed pandas data (dates and numeric series) into a JSON-friendly structure that a charting library can use directly.
3. How long might some of these computations take (especially retraining three ML models)? Does that change anything about how you structure the endpoint (e.g., should you add a generous timeout on the frontend side, or consider caching results so the same ticker+date-range request doesn't retrain everything from scratch every single time)?
4. How will you handle cross-origin requests, given your frontend and backend will likely run as separate processes/containers?

### Definition of Done
You can start your API locally and use its interactive documentation page (most frameworks like this auto-generate one) to manually send a test request to each endpoint and get back a sensible JSON response — without yet having built any frontend at all.

### Common Mistakes
- **Recomputing expensive things (like fetching the same yfinance data, or retraining all three ML models) on every single request with no caching**, leading to a frustratingly slow user experience even for repeated identical requests. You don't need a sophisticated caching system on day one — even a simple "remember the last few results in memory" approach is a reasonable starting point.
- **Returning raw pandas objects directly from an endpoint.** Most frameworks won't know how to automatically convert these into JSON — you'll need to explicitly convert dates to strings and numeric series to plain lists (or dictionaries) before returning them.
- **Not handling the "ticker doesn't exist" or "not enough historical data" cases.** Think through what should happen if a user requests a brand-new IPO with only two months of trading history — your feature engineering, which relies on rolling windows spanning up to a year, will fail or produce mostly-NaN results, and your endpoint should communicate that clearly rather than crashing with an obscure error.

---

## Step 15 — Frontend (Streamlit, 4 Tabs)

### Goal
Build the interactive interface: ticker/date inputs, and four tabs corresponding to your four backend endpoints, each with interactive charts.

### What you need conceptually

**The basic shape of a Streamlit app for this project:** some kind of sidebar or top section for global inputs (ticker, date range, maybe the prediction horizon), a row of tabs, and inside each tab, a mix of summary numbers/tables and interactive charts, populated by calling your backend API and rendering whatever it returns.

**Per-tab content, thinking through what each one needs to *show*, not just *contain*:**
- **Baseline tab:** A table of macro predictors and their OOS R² values (ideally with some visual cue distinguishing positive from negative results, since that's the whole point of the WG finding), plus a chart showing how each predictor's cumulative OOS R² evolved through time.
- **ML Forecast tab:** A side-by-side comparison of your three models' OOS R² values, a feature importance chart (most naturally as a bar chart, sorted from most to least important), and the backtest comparison chart (your strategy's cumulative growth curve next to buy-and-hold's).
- **Monte Carlo tab:** Inputs for the user to adjust the time horizon (the spec calls for 30/90/180/365-day options) and possibly the assumed return/volatility, and the resulting fan chart showing percentile bands widening over time.
- **DCF tab:** Sliders for WACC, 5-year growth rate, and terminal growth rate; resulting intrinsic value, margin of safety, and a waterfall-style chart of the valuation components; and a clearly visible disclaimer that this is a simplified educational model, not financial advice.

**Interactivity requirement:** the project spec is explicit that charts should be interactive (zoomable/hoverable), not static images — keep this in mind when choosing your charting approach, since some plotting libraries default to static output and others are interactive by default.

### Design decisions to make yourself
1. How will user inputs (ticker, dates, horizon, slider values) trigger new backend calls? Think through whether you want every single slider movement to immediately trigger a new request (potentially slow and choppy), or whether some tabs should wait for an explicit "run" button press.
2. How will you handle the time a backend call takes (especially the ML tab, which retrains models)? Think about giving the user some visual feedback that something is happening, rather than leaving the screen looking frozen or unresponsive.
3. How will you structure your code so you're not duplicating the same "call this endpoint, handle errors, render this chart" pattern four times with subtle inconsistencies? Even a simple shared pattern (not necessarily a fancy abstraction) will save you maintenance headaches.

### Definition of Done
You can launch the frontend, enter a ticker and date range, and successfully see results populate in all four tabs by making live calls to your running backend — including genuinely interactive charts (you can hover over points, zoom into a date range, etc.) rather than static images.

### Common Mistakes
- **Hardcoding the backend's address in a way that breaks once you containerize things.** Inside Docker, your frontend and backend containers won't be able to reach each other via `localhost` the same way they can when you're running both directly on your own machine — think ahead about how you'll make this address configurable rather than hardcoded.
- **Not handling backend errors gracefully in the UI.** If a user types an invalid ticker or an unreasonable date range, your frontend should show a clear, friendly message rather than a confusing stack trace or a blank, broken-looking page.
- **Forgetting the DCF disclaimer**, given this is an explicit, named requirement in your project spec, not just a nice-to-have.

---

## Step 16 — Dockerize and Deploy

### Goal
Package the backend and frontend so the whole app can be started with a single command, anywhere — your laptop, a teammate's laptop, or a cloud server.

### What you need conceptually

**Why Docker, conceptually:** your app currently depends on a specific set of installed Python packages, a specific Python version, and (for the backend) possibly some system-level libraries. Docker lets you describe all of these dependencies once, in a reproducible recipe, so that "it works on my machine" becomes "it works anywhere this recipe is run."

**Two separate concerns to package:** your backend (FastAPI + all your ML/data libraries) is one self-contained piece; your frontend (Streamlit) is a different self-contained piece, since it has its own, much smaller, set of dependencies. Running them as two separate containers that talk to each other over a network connection is the conventional approach, rather than trying to cram both into a single container.

**What a typical packaging recipe needs to specify, conceptually, for each piece:** which base operating system/Python image to start from, how to install your specific list of dependencies, how to bring in your actual code, and what command should run when the container starts up.

**Coordinating two containers together:** since your frontend needs to reach your backend over a network address, and that address looks different depending on whether you're running things directly on your machine versus inside this multi-container setup, you'll want a way to define both containers together, including how they're networked to each other and in what order they should start up (the backend needs to be ready before the frontend starts trying to call it).

### Design decisions to make yourself
1. What's the minimal base image you need for each container? (Hint: you don't need a full desktop OS image — a slim, Python-specific base image is usually sufficient and meaningfully smaller/faster to build.)
2. How will the frontend know the backend's network address once both are running inside this coordinated multi-container setup, versus when you're just running them directly on your own machine for quick local testing? Think about making this configurable rather than hardcoding one or the other.
3. Once this works locally, where will you actually deploy it? Research a couple of options (lightweight cloud platforms that support multi-container deployments are a reasonable starting point) and think through what changes (if any) your setup needs for a cloud environment versus your local machine — e.g., environment variables, exposed ports, or basic health checks so the platform knows your service is alive.

### Definition of Done
A single command builds and starts both your backend and frontend in their own isolated containers, they can successfully talk to each other, and you can access the running frontend through your browser exactly as you could when running things locally without Docker at all.

### Common Mistakes
- **Forgetting to re-point your frontend's backend URL when moving from "running directly on my machine" to "running inside Docker."** This is an extremely common stumbling block — the networking address that works in one context typically does not work unchanged in the other.
- **Installing unnecessary system packages or leaving build tools in your final image**, bloating the image size and slowing down both build times and deployments. It's fine to not over-optimize this on your first pass, but be aware it's a thing to revisit.
- **Not testing the health/availability of the backend before the frontend tries to use it.** If the frontend container starts and immediately tries to call an API that's still in the middle of starting up, you'll get confusing intermittent failures — coordinate startup order explicitly rather than hoping for the best.

---

## Final Sanity-Check Checklist (Run This Before You Trust Any Result)

Go through this list any time a result looks "too good," and ideally as a matter of habit before you consider any step really finished:

- [ ] Every feature, at any given date, only uses information that would genuinely have been available on that date.
- [ ] The prediction target is a forward-looking quantity, correctly aligned so it sits on the row representing "the day this forecast would have been made," not the day the outcome became known.
- [ ] Any scaling/normalization was fit exclusively on training data, never on validation or test data.
- [ ] Train/validation/test splits are in strict chronological order, with zero shuffling and zero overlap.
- [ ] Hyperparameters for every model were chosen using validation performance only; the test set was used for evaluation exactly once, at the end.
- [ ] Reported out-of-sample R² values are small (often even negative for the WG macro baselines, and only modestly positive — well under a few percent — for the ML models on stock-level monthly-ish horizons). Suspiciously large positive numbers are a leakage red flag, not a reason to celebrate.
- [ ] The backtest is computed strictly over the test period, using only signals that would have been knowable at each point in time.
- [ ] The Monte Carlo simulation's parameters are explicitly traceable back to your ML model's outputs (not silently falling back to a generic historical average without you realizing it).
- [ ] The DCF tab clearly communicates its assumptions and includes the required "this is not financial advice" disclaimer.

If you can confidently check every box above, you've built something genuinely sound — and quite a bit more rigorous than a lot of "stock predictor" toy projects floating around online.
