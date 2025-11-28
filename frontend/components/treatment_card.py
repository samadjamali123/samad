"""
Treatment Card Component

This component displays comprehensive treatment recommendations with priority ranking,
cost analysis, and interactive elements for different treatment approaches.
"""

import streamlit as st
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class TreatmentPriority(Enum):
    """Treatment priority levels"""
    ORGANIC = "organic"
    CHEMICAL = "chemical"
    PREVENTION = "prevention"
    PROFESSIONAL = "professional"

@dataclass
class TreatmentMethod:
    """Treatment method data structure"""
    name: str
    application: str
    dosage: str
    safety: str
    waiting_period: Optional[str]
    effectiveness: str
    cost_range: str
    environmental_impact: Optional[str] = None

@dataclass
class TreatmentCategory:
    """Treatment category data structure"""
    priority: int
    methods: List[TreatmentMethod]
    cultural_practices: List[str]
    effectiveness_summary: str

@dataclass
class ProfessionalConsultation:
    """Professional consultation data"""
    when_to_consult: List[str]
    contact_resources: List[str]
    yield_loss_thresholds: List[str]

class TreatmentCard:
    """Comprehensive treatment card with prioritized recommendations"""

    def __init__(self, disease_name: str, api_client):
        """Initialize treatment card with disease name and API client"""
        self.disease_name = disease_name
        self.api_client = api_client
        self.treatment_data = None
        self.selected_priority = None
        self.selected_method = None
        self.calculator_data = {
            'garden_size': 100,  # sq ft
            'plants_count': 10,
            'severity_level': 'moderate',
            'budget': 50,
            'preference': 'organic'
        }

    def display(self):
        """Display comprehensive treatment recommendations"""
        if not self.disease_name:
            st.warning("No disease selected for treatment recommendations.")
            return

        # Load treatment data
        self._load_treatment_data()

        if not self.treatment_data:
            st.error(f"No treatment data available for {self.disease_name}")
            return

        # Display main sections
        self._display_header()
        self._display_treatment_calculator()
        self._display_organic_treatments()
        self._display_chemical_treatments()
        self._display_prevention_strategies()
        self._display_professional_consultation()
        self._display_treatment_summary()

    def _load_treatment_data(self):
        """Load treatment data from API or database"""
        try:
            if hasattr(self.api_client, 'get_disease_info'):
                # Try API first
                response = self.api_client.get_disease_info(self.disease_name, include_treatments=True)
                if response and response.get('success', False):
                    self.treatment_data = response.get('treatments')
                    return
        except Exception as e:
            st.warning(f"Could not load treatment data from API: {str(e)}")

        # Fallback to static data (for demo purposes)
        self.treatment_data = self._get_fallback_treatment_data()

    def _get_fallback_treatment_data(self) -> Dict[str, Any]:
        """Get fallback treatment data for demonstration"""
        if "early blight" in self.disease_name.lower():
            return {
                'disease_id': 'tomato_early_blight',
                'organic': {
                    'priority': 1,
                    'methods': [
                        {
                            'name': 'Copper Fungicide',
                            'application': 'Spray every 7-10 days, thoroughly covering leaf undersides',
                            'dosage': '2 tablespoons copper fungicide per gallon water',
                            'safety': 'Wear gloves, avoid inhalation, wash hands after use',
                            'effectiveness': '70-80% effective when applied preventively',
                            'cost_range': 'Low ($10-20 per treatment)',
                            'environmental_impact': 'Copper can accumulate in soil, use sparingly'
                        },
                        {
                            'name': 'Neem Oil',
                            'application': 'Spray leaves thoroughly, including undersides',
                            'dosage': '2 tablespoons neem oil + 1 teaspoon gentle soap per gallon water',
                            'safety': 'Non-toxic, safe for edible plants, wash produce before eating',
                            'effectiveness': '60-70% effective for mild infections',
                            'cost_range': 'Very low ($5-10 per treatment)',
                            'environmental_impact': 'Biodegradable, safe for beneficial insects'
                        }
                    ],
                    'cultural_practices': [
                        'Remove and destroy affected leaves immediately',
                        'Improve air circulation around plants through proper spacing',
                        'Water at base of plants, avoid overhead irrigation',
                        'Apply mulch to prevent soil splash onto leaves'
                    ],
                    'effectiveness_summary': '70-80% effective when combined with cultural practices'
                },
                'chemical': {
                    'priority': 2,
                    'methods': [
                        {
                            'name': 'Chlorothalonil',
                            'application': 'Spray at first sign of disease, repeat every 7-10 days',
                            'dosage': '2 teaspoons per gallon water',
                            'safety': 'Wear gloves, mask, eye protection, wash hands thoroughly',
                            'waiting_period': '0 days before harvest (check label)',
                            'effectiveness': '85-95% effective for established infections',
                            'cost_range': 'Medium ($15-30 per treatment)',
                            'environmental_impact': 'Broad-spectrum fungicide, can affect beneficial organisms'
                        }
                    ],
                    'effectiveness_summary': '85-95% effective but requires safety precautions'
                },
                'prevention': {
                    'crop_rotation': 'Avoid tomatoes in same soil for 3 years',
                    'plant_spacing': '24-36 inches between plants for good air circulation',
                    'resistant_varieties': ['Celebrity', 'Mountain Fresh', 'Defender', 'Iron Lady'],
                    'monitoring': 'Check leaves twice weekly during humid periods'
                },
                'professional_consultation': {
                    'when_to_consult': [
                        'Disease affects >30% of plant foliage',
                        'Rapid spread despite treatment efforts',
                        'Yield loss exceeds 20%',
                        'Multiple diseases present simultaneously'
                    ],
                    'contact_resources': [
                        'Local agricultural extension office',
                        'University plant pathology clinic',
                        'Certified crop advisor',
                        'Master gardener program'
                    ]
                }
            }
        return {}

    def _display_header(self):
        """Display treatment card header"""
        st.markdown(f"""
        <div class="treatment-header">
            <div class="treatment-title">
                <h2>💊 Treatment Recommendations</h2>
                <p>Comprehensive treatment plan for <strong>{self.disease_name}</strong></p>
            </div>
            <div class="treatment-priority">
                <span class="priority-badge organic">Priority 1: Organic</span>
                <span class="priority-badge chemical">Priority 2: Chemical</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick summary stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🌿 Organic Methods", "2 Available", delta="Low cost")

        with col2:
            st.metric("⚗️ Chemical Methods", "1 Available", delta="High effectiveness")

        with col3:
            st.metric("🔒 Prevention", "4 Strategies", delta="Proactive")

        with col4:
            st.metric("👨‍🔬 Professional Help", "When to consult", delta="Expert guidance")

    def _display_treatment_calculator(self):
        """Display interactive treatment calculator"""
        with st.expander("🧮 Treatment Calculator", expanded=False):
            st.markdown("**Customize treatment recommendations based on your garden**")

            col1, col2 = st.columns(2)

            with col1:
                # Garden parameters
                st.markdown("### Garden Parameters")
                self.calculator_data['garden_size'] = st.number_input(
                    "Garden Size (sq ft)",
                    min_value=10,
                    max_value=10000,
                    value=self.calculator_data['garden_size'],
                    help="Total garden area in square feet"
                )

                self.calculator_data['plants_count'] = st.number_input(
                    "Number of Plants",
                    min_value=1,
                    max_value=1000,
                    value=self.calculator_data['plants_count'],
                    help="Total number of affected plants"
                )

                self.calculator_data['severity_level'] = st.selectbox(
                    "Disease Severity",
                    options=['mild', 'moderate', 'severe'],
                    index=1,  # Default to moderate
                    help="Severity of disease symptoms"
                )

            with col2:
                # Treatment preferences
                st.markdown("### Treatment Preferences")
                self.calculator_data['budget'] = st.number_input(
                    "Budget ($)",
                    min_value=5,
                    max_value=500,
                    value=self.calculator_data['budget'],
                    help="Maximum budget for treatments"
                )

                self.calculator_data['preference'] = st.selectbox(
                    "Treatment Preference",
                    options=['organic', 'chemical', 'mixed'],
                    index=0,
                    help="Preferred treatment approach"
                )

                # Calculate recommendations
                if st.button("🔢 Calculate Recommendations", use_container_width=True):
                    self._calculate_and_display_recommendations()

    def _calculate_and_display_recommendations(self):
        """Calculate and display personalized recommendations"""
        st.success("🧮 Calculating personalized recommendations...")

        # Calculate based on calculator data
        garden_size = self.calculator_data['garden_size']
        plants_count = self.calculator_data['plants_count']
        severity = self.calculator_data['severity_level']
        budget = self.calculator_data['budget']
        preference = self.calculator_data['preference']

        # Severity multiplier for treatment amounts
        severity_multiplier = {'mild': 1.0, 'moderate': 1.5, 'severe': 2.0}[severity]

        # Calculate amounts needed
        organic_treatments = self.treatment_data.get('organic', {}).get('methods', [])
        chemical_treatments = self.treatment_data.get('chemical', {}).get('methods', [])

        # Display calculated recommendations
        st.markdown("### 📊 Calculated Treatment Plan")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Recommended Treatment Mix**")

            if preference in ['organic', 'mixed']:
                for i, treatment in enumerate(organic_treatments[:2]):  # Show top 2 organic
                    base_cost = 10 if i == 0 else 5  # Example costs
                    total_cost = (base_cost * plants_count * severity_multiplier) / 10

                    if total_cost <= budget:
                        st.success(f"✅ **{treatment['name']}**")
                        st.markdown(f"• Cost: ${total_cost:.2f}")
                        st.markdown(f"• Coverage: {min(plants_count, 50)} plants")
                    else:
                        st.info(f"💰 **{treatment['name']}**")
                        st.markdown(f"• Cost: ${total_cost:.2f} (over budget)")

            if preference in ['chemical', 'mixed']:
                for treatment in chemical_treatments[:1]:  # Show top 1 chemical
                    base_cost = 20
                    total_cost = (base_cost * plants_count * severity_multiplier) / 10

                    if total_cost <= budget:
                        st.warning(f"⚗️ **{treatment['name']}**")
                        st.markdown(f"• Cost: ${total_cost:.2f}")
                        st.markdown(f"• Coverage: {min(plants_count, 100)} plants")
                    else:
                        st.markdown(f"💰 **{treatment['name']}**")
                        st.markdown(f"• Cost: ${total_cost:.2f} (over budget)")

        with col2:
            st.markdown("**Application Schedule**")

            # Generate application schedule
            if preference in ['organic', 'mixed']:
                st.markdown("#### Organic Treatments")
                st.markdown("- **Week 1**: Initial application")
                st.markdown("- **Week 2**: Second application")
                st.markdown("- **Week 4**: Third application")
                st.markdown("- **Week 6**: Fourth application")
                st.markdown("- **Ongoing**: Every 2 weeks during humid conditions")

            if preference in ['chemical', 'mixed']:
                st.markdown("#### Chemical Treatments")
                st.markdown("- **Day 1**: Immediate application")
                st.markdown("- **Day 7**: Second application")
                st.markdown("- **Day 14**: Third application")
                st.markdown("- **Prevention**: Every 10-14 days")

            # Storage and safety info
            st.markdown("---")
            st.markdown("**Storage & Safety**")
            st.markdown("- Store in cool, dry place")
            st.markdown("- Keep out of reach of children and pets")
            st.markdown("- Follow label instructions carefully")
            st.markdown("- Wear protective equipment during application")

    def _display_organic_treatments(self):
        """Display organic treatment options"""
        organic_data = self.treatment_data.get('organic', {})
        if not organic_data:
            return

        with st.expander("🌿 Organic Treatments (Priority 1)", expanded=True):
            # Cultural practices first
            cultural_practices = organic_data.get('cultural_practices', [])
            if cultural_practices:
                st.markdown("### 🌱 Cultural Practices")
                st.markdown("**Non-chemical preventive measures that reduce disease spread:**")

                practice_cols = st.columns(min(len(cultural_practices), 2))

                for i, practice in enumerate(cultural_practices):
                    with practice_cols[i % len(practice_cols)]:
                        st.markdown(f"""
                        <div class="practice-card">
                            <div class="practice-icon">🌿</div>
                            <div class="practice-text">{practice}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Organic treatment methods
            methods = organic_data.get('methods', [])
            if methods:
                st.markdown("### 💊 Organic Treatment Methods")

                effectiveness_summary = organic_data.get('effectiveness_summary', '')
                if effectiveness_summary:
                    st.info(f"📊 **Overall Effectiveness:** {effectiveness_summary}")

                for i, method in enumerate(methods):
                    self._display_treatment_method(method, "organic", i + 1)

    def _display_chemical_treatments(self):
        """Display chemical treatment options"""
        chemical_data = self.treatment_data.get('chemical', {})
        if not chemical_data:
            return

        with st.expander("⚗️ Chemical Treatments (Priority 2)", expanded=False):
            # Warning banner
            st.markdown("""
            <div class="chemical-warning">
                <div class="warning-icon">⚠️</div>
                <div class="warning-content">
                    <h4>Chemical Treatment Notice</h4>
                    <p>Chemical treatments require proper safety precautions and should be used according to label instructions.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            methods = chemical_data.get('methods', [])
            if methods:
                effectiveness_summary = chemical_data.get('effectiveness_summary', '')
                if effectiveness_summary:
                    st.warning(f"📊 **Overall Effectiveness:** {effectiveness_summary}")

                for i, method in enumerate(methods):
                    self._display_treatment_method(method, "chemical", i + 1)

    def _display_treatment_method(self, method: Dict[str, Any], category: str, number: int):
        """Display individual treatment method details"""
        # Extract method data
        name = method.get('name', 'Unknown Treatment')
        application = method.get('application', 'Not specified')
        dosage = method.get('dosage', 'Not specified')
        safety = method.get('safety', 'Follow standard safety procedures')
        effectiveness = method.get('effectiveness', 'Effectiveness not specified')
        cost_range = method.get('cost_range', 'Cost not specified')
        waiting_period = method.get('waiting_period')
        environmental_impact = method.get('environmental_impact')

        # Color coding based on category
        category_colors = {
            'organic': {'bg': '#E8F5E8', 'border': '#4CAF50', 'icon': '🌿'},
            'chemical': {'bg': '#FFEAEA', 'border': '#F44336', 'icon': '⚗️'}
        }

        colors = category_colors.get(category, {'bg': '#F5F5F5', 'border': '#999999', 'icon': '💊'})

        st.markdown(f"""
        <div class="treatment-method" style="background-color: {colors['bg']}; border-left: 4px solid {colors['border']};">
            <div class="method-header">
                <div class="method-number">{colors['icon']} {number}</div>
                <div class="method-title">{name}</div>
            </div>

            <div class="method-content">
                <div class="method-section">
                    <h4>📋 Application</h4>
                    <p>{application}</p>
                </div>

                <div class="method-section">
                    <h4>📏 Dosage</h4>
                    <p><strong>{dosage}</strong></p>
                </div>

                <div class="method-section">
                    <h4>🛡️ Safety</h4>
                    <p>{safety}</p>
                </div>

                <div class="method-grid">
                    <div class="method-item">
                        <h5>🎯 Effectiveness</h5>
                        <p>{effectiveness}</p>
                    </div>
                    <div class="method-item">
                        <h5>💰 Cost</h5>
                        <p>{cost_range}</p>
                    </div>
        """, unsafe_allow_html=True)

        if waiting_period:
            st.markdown(f"""
            <div class="method-item">
                <h5>⏰ Waiting Period</h5>
                <p>{waiting_period}</p>
            </div>
            """, unsafe_allow_html=True)

        if environmental_impact:
            st.markdown(f"""
            <div class="method-item full-width">
                <h5>🌍 Environmental Impact</h5>
                <p>{environmental_impact}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _display_prevention_strategies(self):
        """Display disease prevention strategies"""
        prevention_data = self.treatment_data.get('prevention', {})
        if not prevention_data:
            return

        with st.expander("🔒 Prevention Strategies", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔄 Crop Management")

                crop_rotation = prevention_data.get('crop_rotation')
                if crop_rotation:
                    st.markdown(f"""
                    <div class="prevention-item">
                        <div class="prevention-icon">🔄</div>
                        <div class="prevention-content">
                            <h4>Crop Rotation</h4>
                            <p>{crop_rotation}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                plant_spacing = prevention_data.get('plant_spacing')
                if plant_spacing:
                    st.markdown(f"""
                    <div class="prevention-item">
                        <div class="prevention-icon">📏</div>
                        <div class="prevention-content">
                            <h4>Plant Spacing</h4>
                            <p>{plant_spacing}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("### 🌱 Variety Selection")

                resistant_varieties = prevention_data.get('resistant_varieties', [])
                if resistant_varieties:
                    st.markdown("**Resistant Varieties:**")
                    for variety in resistant_varieties:
                        st.markdown(f"- ✅ {variety}")

                monitoring = prevention_data.get('monitoring')
                if monitoring:
                    st.markdown("---")
                    st.markdown("### 🔍 Monitoring")

                    st.markdown(f"""
                    <div class="prevention-item">
                        <div class="prevention-icon">🔍</div>
                        <div class="prevention-content">
                            <h4>Regular Monitoring</h4>
                            <p>{monitoring}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Additional prevention tips
            st.markdown("---")
            st.markdown("### 💡 Additional Prevention Tips")
            st.markdown("""
            <div class="prevention-grid">
                <div class="prevention-tip">
                    <div class="tip-icon">💧</div>
                    <div class="tip-text">
                        <strong>Proper Watering</strong><br>
                        Water at base of plants, avoid overhead irrigation
                    </div>
                </div>

                <div class="prevention-tip">
                    <div class="tip-icon">🌬️</div>
                    <div class="tip-text">
                        <strong>Good Air Circulation</strong><br>
                        Ensure adequate spacing between plants
                    </div>
                </div>

                <div class="prevention-tip">
                    <div class="tip-icon">🧹</div>
                    <div class="tip-text">
                        <strong>Sanitation</strong><br>
                        Clean tools and remove plant debris
                    </div>
                </div>

                <div class="prevention-tip">
                    <div class="tip-icon">🌤</div>
                    <div class="tip-text">
                        <strong>Optimal Conditions</strong><br>
                        Maintain appropriate humidity and temperature
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def _display_professional_consultation(self):
        """Display professional consultation guidelines"""
        prof_data = self.treatment_data.get('professional_consultation', {})
        if not prof_data:
            return

        with st.expander("👨‍🔬 Professional Consultation", expanded=False):
            st.markdown("### 📞 When to Consult a Professional")

            when_to_consult = prof_data.get('when_to_consult', [])
            if when_to_consult:
                for i, condition in enumerate(when_to_consult, 1):
                    severity = "🔴" if i <= 2 else "🟡" if i <= 3 else "🟢"
                    st.markdown(f"""
                    <div class="consultation-condition">
                        <div class="condition-severity">{severity}</div>
                        <div class="condition-text">{condition}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Contact resources
            contact_resources = prof_data.get('contact_resources', [])
            if contact_resources:
                st.markdown("### 🏥 Contact Resources")
                st.markdown("**For professional help and expert advice:**")

                for resource in contact_resources:
                    st.markdown(f"- 📞 {resource}")

            # Emergency indicators
            st.markdown("---")
            st.markdown("### 🚨 Emergency Situations")
            st.markdown("""
            <div class="emergency-alert">
                <div class="emergency-icon">🚨</div>
                <div class="emergency-content">
                    <h4>Seek Immediate Help If:</h4>
                    <ul>
                        <li>Disease is spreading rapidly (affects >50% of plants)</li>
                        <li>Multiple plants are dying simultaneously</li>
                        <li>Unusual symptoms don't match common diseases</li>
                        <li>Economic losses are significant (>25% of crop)</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def _display_treatment_summary(self):
        """Display comprehensive treatment summary"""
        with st.expander("📋 Treatment Summary", expanded=False):
            st.markdown("### 💊 Recommended Action Plan")

            # Create timeline
            st.markdown("""
            <div class="treatment-timeline">
                <div class="timeline-item immediate">
                    <div class="timeline-marker">🔥</div>
                    <div class="timeline-content">
                        <h4>Immediate (Today)</h4>
                        <p>• Remove affected leaves<br>
                           • Apply first organic treatment<br>
                           • Improve air circulation</p>
                    </div>
                </div>

                <div class="timeline-item short-term">
                    <div class="timeline-marker">📅</div>
                    <div class="timeline-content">
                        <h4>Short Term (1-2 weeks)</h4>
                        <p>• Continue organic treatments<br>
                           • Monitor spread carefully<br>
                           • Apply preventive measures</p>
                    </div>
                </div>

                <div class="timeline-item ongoing">
                    <div class="timeline-marker">🔄</div>
                    <div class="timeline-content">
                        <h4>Ongoing (2+ weeks)</h4>
                        <p>• Regular monitoring<br>
                           • Seasonal prevention<br>
                           • Consider resistant varieties</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Cost summary
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 💰 Estimated Cost Analysis")

                # Calculate estimated costs
                organic_methods = self.treatment_data.get('organic', {}).get('methods', [])
                chemical_methods = self.treatment_data.get('chemical', {}).get('methods', [])

                organic_cost = 15 * len(organic_methods)  # Average cost per method
                chemical_cost = 25 * len(chemical_methods)

                st.markdown(f"""
                <div class="cost-summary">
                    <div class="cost-item">
                        <span class="cost-label">Organic Treatments:</span>
                        <span class="cost-value">${organic_cost}</span>
                    </div>
                    <div class="cost-item">
                        <span class="cost-label">Chemical Treatments:</span>
                        <span class="cost-value">${chemical_cost}</span>
                    </div>
                    <div class="cost-item total">
                        <span class="cost-label">Total Range:</span>
                        <span class="cost-value">${organic_cost + chemical_cost}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("### 📊 Success Probability")

                st.markdown("""
                <div class="success-probability">
                    <div class="probability-item high">
                        <div class="probability-bar" style="width: 75%;"></div>
                        <span>Organic Only: 70-80%</span>
                    </div>
                    <div class="probability-item">
                        <div class="probability-bar" style="width: 90%;"></div>
                        <span>Integrated Approach: 85-95%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Action buttons
            st.markdown("---")
            st.markdown("### 🎯 Quick Actions")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("📞 Contact Expert", key="contact_expert", use_container_width=True):
                    st.info("Contact your local agricultural extension office or plant clinic for personalized advice.")

            with col2:
                if st.button("📄 Print Summary", key="print_summary", use_container_width=True):
                    st.success("Treatment summary prepared for printing!")

            with col3:
                if st.button("📤 Share Results", key="share_results", use_container_width=True):
                    st.success("Ready to share with your farming community!")

            with col4:
                if st.button("🔄 Re-analyze", key="reanalyze", use_container_width=True):
                    st.info("Upload a new image for another analysis.")

    def get_treatment_summary(self) -> Dict[str, Any]:
        """Get treatment summary for export or sharing"""
        if not self.treatment_data:
            return {}

        return {
            'disease_name': self.disease_name,
            'organic_methods': len(self.treatment_data.get('organic', {}).get('methods', [])),
            'chemical_methods': len(self.treatment_data.get('chemical', {}).get('methods', [])),
            'prevention_strategies': len(self.treatment_data.get('prevention', {})),
            'calculator_data': self.calculator_data,
            'priority_order': ['organic', 'chemical', 'prevention'],
            'last_updated': time.time()
        }