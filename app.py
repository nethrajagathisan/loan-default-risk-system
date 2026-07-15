import sys
import os
from pathlib import Path

# Get the repository root directory (go up from dashboard/ to repo root)
repo_root = Path(__file__).parent
os.chdir(repo_root)  # Change working directory to repo root
sys.path.insert(0, str(repo_root))  # Add repo root to Python path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import os
try:
    from src.explainability import LoanExplainer, load_test_data
except Exception as e:
    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    st.error(f"Startup import error:\n```\n{tb_str}\n```")
    st.stop()

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk system",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .approved {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
    }
    
    .declined {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load model and sample data with caching"""
    explainer = None
    X_sample = None
    y_sample = None

    # Step 1: Load model (always works if model files are in models/)
    try:
        with st.spinner("Loading model and initializing SHAP explainer..."):
            explainer = LoanExplainer()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure model files are available in the 'models/' directory")
        return None, None, None

    # Step 2: Load processed portfolio data (optional — enables the portfolio pages)
    try:
        test_path = "data/processed/model_df.parquet"
        X_sample, y_sample = load_test_data(test_path)

        sample_size = min(5000, len(X_sample))
        sample_indices = X_sample.sample(sample_size, random_state=42).index
        X_sample = X_sample.loc[sample_indices].reset_index(drop=True)
        if y_sample is not None:
            y_sample = y_sample.loc[sample_indices].reset_index(drop=True)
    except Exception as e:
        st.warning(f"⚠️ Could not load test data: {str(e)}")
        st.info("📊 Individual loan risk assessment is still fully available! Navigate to '🔍 Risk Assessment' in the sidebar.")
        X_sample = None
        y_sample = None

    return explainer, X_sample, y_sample


def create_kpi_metrics(explainer, X_data):
    """Create KPI metrics for executive dashboard"""
    risk_dist = explainer.get_risk_distribution(X_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Approval Rate</h3>
            <h2>{risk_dist['approval_rate']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Avg Risk Score</h3>
            <h2>{risk_dist['mean_risk']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>High Risk Rate</h3>
            <h2>{risk_dist['high_risk_rate']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Current Threshold</h3>
            <h2>{explainer.threshold:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    return risk_dist


def show_individual_assessment(explainer, X_sample):

    # Model feature layout: prefer sample data, fall back to saved feature names
    if X_sample is not None:
        feature_cols = list(X_sample.columns)
    elif explainer.feature_names is not None:
        feature_cols = list(explainer.feature_names)
    else:
        st.error("Model feature names unavailable. Re-run `python src/model_training.py`.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Applicant Information")

        # BASIC APPLICANT INFO
        st.markdown("**Personal & Employment Details**")

        age_years = st.number_input(
            "Applicant Age (years)",
            min_value=18, max_value=100,
            value=35,
            step=1,
            help="Applicant's age at loan disbursal"
        )

        employment_type = st.selectbox(
            "Employment Type",
            options=["Unknown", "Salaried", "Self-employed"],
            index=1,
            help="Applicant's employment category"
        )

        st.markdown("**KYC Documents Submitted**")
        kyc_col1, kyc_col2 = st.columns(2)
        with kyc_col1:
            aadhaar = st.checkbox("Aadhaar", value=True)
            pan = st.checkbox("PAN", value=False)
            voter_id = st.checkbox("Voter ID", value=False)
        with kyc_col2:
            driving_licence = st.checkbox("Driving Licence", value=False)
            passport = st.checkbox("Passport", value=False)

        # CREDIT BUREAU PROFILE
        st.markdown("**Credit Bureau Profile**")

        no_bureau_history = st.checkbox(
            "No bureau history (first-time borrower)",
            value=False,
            help="Tick if the applicant has no credit bureau record"
        )

        bureau_score = st.number_input(
            "Bureau / CIBIL Score",
            min_value=300, max_value=900,
            value=700,
            step=5,
            help="Credit bureau score (300-900 range)",
            disabled=no_bureau_history
        )

        bureau_grade = create_bureau_grade_input(disabled=no_bureau_history)

        total_accounts = st.number_input(
            "Total Bureau Accounts",
            value=4, step=1, min_value=0,
            help="Total loan accounts reported at the credit bureau"
        )

        active_accounts = st.number_input(
            "Active Loan Accounts",
            value=2, step=1, min_value=0,
            help="Currently active loan accounts"
        )

        overdue_accounts = st.number_input(
            "Overdue Accounts",
            value=0, step=1, min_value=0,
            help="Accounts currently past due"
        )

        delinq_accounts_6m = st.number_input(
            "Delinquent Accounts (Last 6 Months)",
            value=0, step=1, min_value=0,
            help="Accounts that went delinquent in the past 6 months"
        )

        credit_inquiries = st.number_input(
            "Bureau Inquiries",
            value=1, step=1, min_value=0,
            help="Number of recent credit inquiries"
        )

        credit_history_years = st.number_input(
            "Credit History Length (years)",
            value=3.0, step=0.5, min_value=0.0,
            help="Time since the applicant's first credit account"
        )

        new_accounts_6m = st.number_input(
            "New Accounts Opened (Last 6 Months)",
            value=0, step=1, min_value=0,
            help="Loan accounts opened in the past 6 months"
        )

        balances_unknown = st.checkbox("Bureau balance details unknown", key="bal_unknown")

        outstanding_balance = st.number_input(
            "Current Outstanding Balance (₹)",
            value=100000, step=10000, min_value=0,
            help="Total outstanding balance across bureau accounts",
            disabled=balances_unknown
        )

        sanctioned_credit = st.number_input(
            "Total Sanctioned Credit (₹)",
            value=200000, step=10000, min_value=0,
            help="Total credit sanctioned across bureau accounts",
            disabled=balances_unknown
        )

        existing_emi = st.number_input(
            "Existing Monthly Obligations / EMIs (₹)",
            value=3000, step=500, min_value=0,
            help="Current monthly instalment obligations"
        )

        # LOAN DETAILS
        st.markdown("**Loan Details**")

        loan_amount = st.number_input(
            "Loan Amount to Disburse (₹)",
            value=60000, step=5000, min_value=1000,
            help="Amount of loan to be disbursed"
        )

        asset_cost = st.number_input(
            "Asset / Vehicle Cost (₹)",
            value=80000, step=5000, min_value=1000,
            help="Total cost of the financed asset"
        )

        ltv = min(loan_amount / max(asset_cost, 1) * 100, 100.0)
        st.caption(f"Loan-to-Value (LTV): **{ltv:.1f}%**")
    
    with col2:
        st.subheader("📊 Risk Assessment Results")
        
        if st.button("🔍 Analyze Loan Application", type="primary", use_container_width=True):
            try:
                # Convert user inputs to model features
                model_input = convert_user_inputs_to_model_features_flexible(
                    feature_cols=feature_cols,
                    age_years=age_years,
                    employment_type=employment_type,
                    kyc_docs={
                        'aadhar_flag': aadhaar,
                        'pan_flag': pan,
                        'voterid_flag': voter_id,
                        'driving_flag': driving_licence,
                        'passport_flag': passport,
                    },
                    no_bureau_history=no_bureau_history,
                    bureau_score=bureau_score,
                    bureau_grade=bureau_grade,
                    total_accounts=total_accounts,
                    active_accounts=active_accounts,
                    overdue_accounts=overdue_accounts,
                    delinq_accounts_6m=delinq_accounts_6m,
                    credit_inquiries=credit_inquiries,
                    credit_history_years=credit_history_years,
                    new_accounts_6m=new_accounts_6m,
                    balances_unknown=balances_unknown,
                    outstanding_balance=outstanding_balance,
                    sanctioned_credit=sanctioned_credit,
                    existing_emi=existing_emi,
                    loan_amount=loan_amount,
                    asset_cost=asset_cost,
                    ltv=ltv,
                )
                
                # Get prediction
                pred_info = explainer.predict_loan(model_input)
                
                # Display results
                decision_color = "🟢" if pred_info['decision'] == 'Approved' else "🔴"
                risk_level = get_risk_level(pred_info['calibrated_probability'])
                
                # Main result
                st.markdown(f"""
                ## {decision_color} **{pred_info['decision']}**
                
                ### 📈 **Risk Assessment:**
                - **Default Probability:** {pred_info['calibrated_probability']:.1%}
                - **Risk Level:** {risk_level}
                - **Decision Confidence:** {pred_info.get('confidence', 0.85):.1%}
                """)
                
                # Risk breakdown
                col_risk1, col_risk2 = st.columns(2)
                
                with col_risk1:
                    # Risk gauge
                    fig_gauge = create_risk_gauge(pred_info['calibrated_probability'], pred_info['threshold'])
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col_risk2:
                    # Key factors (only show if values were provided)
                    factors_text = "**Key Decision Factors:**\n"
                    if no_bureau_history:
                        factors_text += "- Bureau History: None (first-time borrower) ⚠️\n"
                    else:
                        factors_text += f"- Bureau Score: {bureau_score} {'✅' if bureau_score >= 700 else '❌'}\n"
                        if bureau_grade is not None:
                            factors_text += f"- Bureau Risk Grade: {bureau_grade[0]} {'✅' if bureau_grade[1] <= 5 else '❌'}\n"
                    factors_text += f"- Loan-to-Value: {ltv:.0f}% {'✅' if ltv <= 85 else '❌'}\n"
                    factors_text += f"- Overdue Accounts: {overdue_accounts} {'✅' if overdue_accounts == 0 else '❌'}\n"
                    factors_text += f"- Delinquencies (6M): {delinq_accounts_6m} {'✅' if delinq_accounts_6m == 0 else '❌'}\n"
                    if employment_type != "Unknown":
                        factors_text += f"- Employment: {employment_type} ✅\n"

                    st.markdown(factors_text)
                
                # SHAP explanation
                with st.expander("📊 View Detailed Risk Analysis"):
                    try:
                        fig_shap, _, _ = explainer.explain_single_loan(model_input)
                        st.pyplot(fig_shap, clear_figure=True)
                    except Exception as shap_error:
                        st.warning(f"Could not generate SHAP explanation: {str(shap_error)}")
                
            except Exception as e:
                st.error(f"Error in risk assessment: {str(e)}")
                with st.expander("Debug Information"):
                    st.write(f"Error details: {str(e)}")

# HELPER FOR BUREAU RISK GRADE

# LTFS bureau grades: A = lowest risk ... M = highest risk
BUREAU_GRADE_DESCRIPTIONS = {
    'A': 'Very Low Risk', 'B': 'Very Low Risk', 'C': 'Very Low Risk', 'D': 'Very Low Risk',
    'E': 'Low Risk', 'F': 'Low Risk', 'G': 'Low Risk',
    'H': 'Medium Risk', 'I': 'Medium Risk',
    'J': 'High Risk', 'K': 'High Risk',
    'L': 'Very High Risk', 'M': 'Very High Risk'
}

def create_bureau_grade_input(disabled=False):
    """Bureau risk grade selector. Returns (letter, ordinal 1-13) or None if unknown."""
    options = ["Unknown"] + [f"{g} - {desc}" for g, desc in BUREAU_GRADE_DESCRIPTIONS.items()]

    selected = st.selectbox(
        "Bureau Risk Grade",
        options=options,
        index=4,  # Default to D - Very Low Risk
        help="Risk grade from the credit bureau (A = lowest risk, M = highest risk)",
        disabled=disabled
    )

    if disabled or selected == "Unknown":
        return None

    letter = selected.split(" ")[0]
    return letter, ord(letter) - ord('A') + 1

# UPDATED CONVERSION FUNCTION

def convert_user_inputs_to_model_features_flexible(feature_cols, age_years, employment_type, kyc_docs,
                                                   no_bureau_history, bureau_score, bureau_grade,
                                                   total_accounts, active_accounts, overdue_accounts,
                                                   delinq_accounts_6m, credit_inquiries, credit_history_years,
                                                   new_accounts_6m, balances_unknown, outstanding_balance,
                                                   sanctioned_credit, existing_emi, loan_amount, asset_cost, ltv):
    """Convert user-friendly inputs to model features with null handling"""

    # Start with every model feature missing (XGBoost handles NaN natively)
    input_df = pd.DataFrame(np.nan, index=[0], columns=feature_cols)

    def set_feat(col, value):
        # Only touch columns the trained model actually knows about
        if col in input_df.columns:
            input_df[col] = value

    # Personal & employment
    set_feat('age_years', age_years)
    set_feat('emp_salaried', 1 if employment_type == 'Salaried' else 0)
    set_feat('emp_self_employed', 1 if employment_type == 'Self-employed' else 0)

    # KYC flags
    for col, ticked in kyc_docs.items():
        set_feat(col, int(ticked))
    set_feat('kyc_docs_count', sum(int(v) for v in kyc_docs.values()))

    # Bureau score & grade
    set_feat('no_bureau_history', int(no_bureau_history))
    set_feat('bureau_not_scored', 0)
    set_feat('bureau_score_missing', int(no_bureau_history))
    if not no_bureau_history:
        set_feat('bureau_score', bureau_score)
        if bureau_grade is not None:
            set_feat('bureau_risk_grade', bureau_grade[1])

    # Bureau accounts
    set_feat('total_accts', total_accounts)
    set_feat('active_accts', active_accounts)
    set_feat('overdue_accts', overdue_accounts)
    set_feat('active_ratio', active_accounts / (total_accounts + 1))
    set_feat('overdue_ratio', overdue_accounts / (active_accounts + 1))

    set_feat('delinquent_accts_in_last_six_months', delinq_accounts_6m)
    set_feat('delinq_flag', int(delinq_accounts_6m > 0))
    set_feat('no_of_inquiries', credit_inquiries)
    set_feat('new_accts_in_last_six_months', new_accounts_6m)
    set_feat('new_credit_ratio', new_accounts_6m / (total_accounts + 1))

    credit_history_months = credit_history_years * 12
    set_feat('credit_history_months', credit_history_months)
    set_feat('age_x_history', age_years * credit_history_months)

    # Balances & obligations
    bureau_util_pct = None
    if not balances_unknown:
        set_feat('current_balance_log', np.log1p(max(outstanding_balance, 0)))
        set_feat('sanctioned_amount_log', np.log1p(max(sanctioned_credit, 0)))
        bureau_util_pct = min(outstanding_balance / (sanctioned_credit + 1), 2) * 100
        set_feat('bureau_util_pct', bureau_util_pct)

    set_feat('monthly_obligation_log', np.log1p(max(existing_emi, 0)))
    set_feat('instal_to_loan_ratio', existing_emi / (loan_amount + 1))

    # Loan terms
    set_feat('disbursed_amount_log', np.log1p(loan_amount))
    set_feat('asset_cost_log', np.log1p(asset_cost))
    set_feat('ltv', ltv)

    # Interaction terms (mirror src/data_processing.py definitions)
    if no_bureau_history:
        set_feat('ltv_score_ratio', ltv / (300 + 1))
    else:
        if bureau_util_pct is not None:
            set_feat('util_score', bureau_util_pct * bureau_score)
        set_feat('ltv_score_ratio', ltv / (bureau_score + 1))
    set_feat('overdue_x_delinq', overdue_accounts * int(delinq_accounts_6m > 0))

    return input_df

# KEEP ALL OTHER HELPER FUNCTIONS THE SAME
def get_risk_level(probability):
    """Convert probability to risk level using balanced industry thresholds"""
    if probability <= 0.15:
        return "🟢 Low Risk"
    elif probability <= 0.30:
        return "🟡 Medium Risk" 
    elif probability <= 0.50:
        return "🟠 High Risk"
    else:
        return "🔴 Very High Risk"

def create_risk_gauge(probability, threshold):
    """Create risk probability gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default Risk %"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 25], 'color': "yellow"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def show_batch_analysis(explainer, X_sample, y_sample=None):
    """Enhanced batch loan risk analysis with optimized sampling"""
    st.header("📊 Portfolio Batch Risk Analysis")
    
    # Sample size configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Portfolio Risk Assessment:** Analyze risk distribution across your loan portfolio 
        to identify concentration risks, optimal approval rates, and expected loss patterns. 
        This analysis helps in portfolio optimization and risk management strategies.
        """)
    
    with col2:
        # Dynamic sample size based on data availability
        max_samples = len(X_sample)
        default_sample = min(10000, max_samples)  # Increased from 100
        
        sample_size = st.slider(
            "Portfolio Sample Size",
            min_value=1000,
            max_value=min(50000, max_samples),
            value=default_sample,
            step=1000,
            help="Larger samples provide more accurate risk estimates but take longer to process"
        )
    
    if st.button("🚀 Run Risk Analysis", type="primary"):
        with st.spinner(f"Analyzing {sample_size:,} loans..."):
            # Optimized sampling strategy
            if sample_size < len(X_sample):
                # Stratified sampling if labels available
                if y_sample is not None:
                    from sklearn.model_selection import train_test_split
                    X_batch, _, y_batch, _ = train_test_split(
                        X_sample, y_sample, 
                        train_size=sample_size/len(X_sample),
                        stratify=y_sample,
                        random_state=42
                    )
                else:
                    # Random sampling
                    X_batch = X_sample.sample(sample_size, random_state=42)
                    y_batch = None
            else:
                X_batch = X_sample
                y_batch = y_sample
            
            # Batch predictions with progress tracking
            progress_bar = st.progress(0)
            
            # Process in chunks to avoid memory issues
            chunk_size = 5000
            all_results = []
            
            for i in range(0, len(X_batch), chunk_size):
                chunk = X_batch.iloc[i:i+chunk_size]
                chunk_results = explainer.batch_predict(chunk)
                all_results.append(chunk_results)
                
                # Update progress
                progress = min((i + chunk_size) / len(X_batch), 1.0)
                progress_bar.progress(progress)
            
            # Combine results
            batch_results = pd.concat(all_results, ignore_index=True)
            progress_bar.empty()
            
            # Display risk distribution analysis
            show_risk_distribution_analysis(batch_results, X_batch, y_batch)

def show_risk_distribution_analysis(batch_results, X_batch, y_batch):
    """Enhanced risk distribution analysis"""
    st.subheader("📈 Portfolio Risk Distribution")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        approval_rate = (batch_results['decision'] == 'Approved').mean()
        st.metric(
            "Approval Rate", 
            f"{approval_rate:.1%}",
            help="Percentage of applications that would be approved"
        )
    
    with col2:
        avg_risk = batch_results['calibrated_probability'].mean()
        st.metric(
            "Average Risk", 
            f"{avg_risk:.2%}",
            help="Average predicted default probability"
        )
    
    with col3:
        high_risk_rate = (batch_results['calibrated_probability'] > 0.3).mean()
        st.metric(
            "High Risk Rate", 
            f"{high_risk_rate:.1%}",
            help="Percentage of high-risk applications (>30% default probability)"
        )
    
    with col4:
        risk_concentration = batch_results['calibrated_probability'].std()
        st.metric(
            "Risk Concentration", 
            f"{risk_concentration:.3f}",
            help="Standard deviation of risk scores (higher = more diverse risk)"
        )
    
    # Risk distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk histogram
        fig_hist = px.histogram(
            batch_results, 
            x='calibrated_probability',
            nbins=50,
            title="Risk Score Distribution",
            labels={'calibrated_probability': 'Default Probability', 'count': 'Number of Loans'}
        )
        fig_hist.add_vline(
            x=batch_results['calibrated_probability'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text="Average Risk"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Decision distribution
        decision_counts = batch_results['decision'].value_counts()
        fig_pie = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title="Approval Decision Distribution",
            color_discrete_map={'Approved': 'green', 'Declined': 'red'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Risk by category breakdown
    if 'risk_category' in batch_results.columns:
        st.subheader("📊 Risk Category Breakdown")
        
        risk_summary = batch_results.groupby('risk_category').agg({
            'calibrated_probability': ['count', 'mean', 'std'],
            'decision': lambda x: (x == 'Approved').mean()
        }).round(3)
        
        risk_summary.columns = ['Count', 'Avg_Risk', 'Risk_Std', 'Approval_Rate']
        risk_summary['Percentage'] = (risk_summary['Count'] / len(batch_results) * 100).round(1)
        
        st.dataframe(risk_summary, use_container_width=True)
    
    # Expected loss calculation if actual outcomes available
    if y_batch is not None:
        st.subheader("💰 Expected vs Actual Performance")
        comparison_df = pd.DataFrame({
            'Predicted_Risk': batch_results['calibrated_probability'],
            'Actual_Default': y_batch.values
        })
        
        # Binned analysis
        comparison_df['Risk_Bin'] = pd.cut(
            comparison_df['Predicted_Risk'], 
            bins=10, 
            labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)]
        )
        
        bin_analysis = comparison_df.groupby('Risk_Bin').agg({
            'Predicted_Risk': 'mean',
            'Actual_Default': 'mean'
        }).reset_index()
        
        fig_calibration = px.scatter(
            bin_analysis,
            x='Predicted_Risk',
            y='Actual_Default',
            title="Model Calibration: Predicted vs Actual Default Rates",
            labels={'Predicted_Risk': 'Predicted Default Rate', 'Actual_Default': 'Actual Default Rate'}
        )
        fig_calibration.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash", color="red"),
            name="Perfect Calibration"
        )
        st.plotly_chart(fig_calibration, use_container_width=True)



def plot_risk_distribution(X_data, explainer):
    """Create risk distribution plot"""
    probabilities = explainer.cal_model.predict_proba(X_data)[:, 1]
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=probabilities,
        nbinsx=50,
        name='Risk Distribution',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Add threshold line
    fig.add_vline(
        x=explainer.threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {explainer.threshold:.3f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Portfolio Risk Distribution",
        xaxis_title="Default Probability",
        yaxis_title="Number of Loans",
        showlegend=False,
        height=400
    )
    
    return fig



def plot_threshold_analysis(explainer, X_data):
    """Create threshold impact analysis"""
    threshold_df = explainer.simulate_threshold_impact(X_data)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Approval Rate vs Threshold', 'Expected Bad Rate vs Threshold'],
        vertical_spacing=0.15
    )
    
    # Approval rate
    fig.add_trace(
        go.Scatter(x=threshold_df['threshold'], y=threshold_df['approval_rate'],
                  mode='lines+markers', name='Approval Rate', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Bad rate
    fig.add_trace(
        go.Scatter(x=threshold_df['threshold'], y=threshold_df['expected_bad_rate'],
                  mode='lines+markers', name='Expected Bad Rate', line=dict(color='red')),
        row=2, col=1
    )
    
    # Current threshold line
    fig.add_vline(x=explainer.threshold, line_dash="dash", line_color="green",
                 annotation_text="Current Threshold")
    
    fig.update_layout(height=500, title_text="Threshold Impact Analysis")
    fig.update_xaxes(title_text="Threshold", row=2, col=1)
    fig.update_yaxes(title_text="Approval Rate", row=1, col=1)
    fig.update_yaxes(title_text="Bad Rate", row=2, col=1)
    
    return fig


def main():
    st.title("🏦 Loan Default Risk System Dashboard")
    st.markdown("Advanced ML-powered loan risk assessment and portfolio optimization")
    
    # Load model and data
    explainer, X_sample, y_sample = load_model_and_data()
    
    if explainer is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Executive Dashboard", "🎯 Risk Assessment", "🔍 Explainability", "⚖️ Portfolio Optimizer"]
    )
    
    if page == "📊 Executive Dashboard":
        st.header("Executive Dashboard")
        
        if X_sample is None:
            st.info("📊 Test dataset is not available. This feature requires portfolio data. Please try the '🔍 Risk Assessment' page for individual loan analysis.")
        else:
            # KPI Metrics
            risk_dist = create_kpi_metrics(explainer, X_sample)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution plot
                risk_fig = plot_risk_distribution(X_sample, explainer)
                st.plotly_chart(risk_fig, use_container_width=True)
                
            with col2:
                # Risk percentiles
                st.subheader("Risk Percentiles")
                perc_df = pd.DataFrame.from_dict(risk_dist['percentiles'], orient='index', columns=['Percentile'])
                perc_df.index.name = 'Level'
                st.dataframe(perc_df, use_container_width=True)
                
                # Portfolio summary
                st.subheader("Portfolio Summary")
                st.metric("Mean Risk", f"{risk_dist['mean_risk']:.3f}")
                st.metric("Risk Std Dev", f"{risk_dist['std_risk']:.3f}")
    
    elif page == "🎯 Risk Assessment":
        st.header("Individual Loan Risk Assessment")
        
        tab1, tab2 = st.tabs(["New Application", "Batch Analysis"])
        
        with tab1:
            show_individual_assessment(explainer, X_sample)
        
        with tab2:
            if X_sample is None:
                st.info("📊 Test dataset is not available. This feature requires portfolio data. Please try the '🔍 Risk Assessment' page for individual loan analysis.")
            else:
                show_batch_analysis(explainer, X_sample, y_sample)
    
    elif page == "🔍 Explainability":
        st.header("Model Explainability & Feature Analysis")
        
        if X_sample is None:
            st.info("📊 Test dataset is not available. This feature requires portfolio data. Please try the '🔍 Risk Assessment' page for individual loan analysis.")
        else:
            tab1, tab2 = st.tabs(["Feature Importance", "Feature Interactions"])
            
            with tab1:
                st.subheader("Global Feature Importance")
                
                # Get feature importance
                importance_df = explainer.get_feature_importance(X_sample, top_n=20)
                
                # Plot importance
                fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                            title="Top 20 Most Important Features")
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show SHAP summary plot
                st.subheader("SHAP Summary Plot")
                try:
                    shap_fig = explainer.create_summary_plot(X_sample.head(500))
                    st.pyplot(shap_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating SHAP plot: {str(e)}")
            
            with tab2:
                st.subheader("Feature Dependence Analysis")
                
                # Feature selection
                selected_feature = st.selectbox(
                    "Select Feature for Dependence Analysis:",
                    options=explainer.feature_names[:20]  # Top 20 for performance
                )
                
                if st.button("Generate Dependence Plot"):
                    try:
                        dep_fig = explainer.create_dependence_plot(X_sample.head(500), selected_feature)
                        st.pyplot(dep_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating dependence plot: {str(e)}")
    
    elif page == "⚖️ Portfolio Optimizer":
        st.header("Portfolio Optimization & Threshold Tuning")
        
        if X_sample is None:
            st.info("📊 Test dataset is not available. This feature requires portfolio data. Please try the '🔍 Risk Assessment' page for individual loan analysis.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current Settings")
                st.metric("Current Threshold", explainer.threshold)
                
                # Threshold slider (for visualization only)
                new_threshold = st.slider(
                    "Explore Threshold Impact:",
                    min_value=0.1, max_value=0.5, 
                    value=float(explainer.threshold), step=0.01
                )
                
                # Calculate impact of new threshold
                probabilities = explainer.cal_model.predict_proba(X_sample)[:, 1]
                new_approval_rate = (probabilities <= new_threshold).mean()
                approved_loans = probabilities[probabilities <= new_threshold]
                new_bad_rate = approved_loans.mean() if len(approved_loans) > 0 else 0
                
                st.metric("Simulated Approval Rate", f"{new_approval_rate:.1%}")
                st.metric("Simulated Bad Rate", f"{new_bad_rate:.3f}")
            
            with col2:
                # Threshold analysis plot
                threshold_fig = plot_threshold_analysis(explainer, X_sample)
                st.plotly_chart(threshold_fig, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Loan Default Risk System Dashboard | Built with Streamlit & XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()