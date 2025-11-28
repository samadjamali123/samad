"""
Disease Database Page - Browse Plant Diseases
"""

import streamlit as st
import time
from utils.api_client import APIClient
from utils.ui_helpers import (
    show_page_header,
    show_empty_state,
    create_disease_card,
    create_search_filters,
    apply_filters_to_diseases,
    format_timestamp,
    show_error_message,
    show_success_message,
    create_disease_severity_chart
)


def show_diseases_page():
    """Display disease database page"""
    show_page_header("🌱 Disease Database", "Comprehensive plant disease information")

    # Check API connection
    if 'api_client' not in st.session_state:
        show_empty_state(
            icon="🔌",
            title="API Not Connected",
            description="Please ensure the backend API is running to access the disease database.",
            action_text="Check Connection",
            action_callback=lambda: st.rerun()
        )
        return

    api_client = st.session_state.api_client

    # Sidebar filters
    create_disease_filters(api_client)

    # Main content
    show_disease_content(api_client)


def create_disease_filters(api_client):
    """Create disease filter sidebar"""
    st.sidebar.markdown("### 🔍 Search Diseases")

    # Search input
    search_query = st.sidebar.text_input(
        "Search diseases...",
        placeholder="Enter disease name, symptoms, or plant type...",
        help="Search by disease name, symptoms, affected plants, or keywords"
    )

    # Get filter options
    categories = get_categories(api_client)
    severity_levels = get_severity_levels(api_client)
    affected_plants = get_affected_plants(api_client)

    # Create filter widgets
    filters = create_search_filters(categories, severity_levels, affected_plants)

    # Filter controls
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", key="refresh_diseases"):
            st.rerun()
    with col2:
        if st.button("🔄 Clear Filters", key="clear_filters"):
            # Clear session filters
            for key in list(st.session_state.keys()):
                if key.startswith('filter_'):
                    del st.session_state[key]
            st.rerun()

    return filters


def get_categories(api_client):
    """Get disease categories from API"""
    try:
        response = api_client.get_disease_categories()
        if response.get('success'):
            return response.get('categories', [])
        return []
    except Exception as e:
        st.error(f"Error loading categories: {str(e)}")
        return []


def get_severity_levels(api_client):
    """Get severity levels from API"""
    try:
        response = api_client.get_all_diseases()
        if response.get('success'):
            diseases = response.get('diseases', [])
            severities = list(set(d.get('severity', 'Unknown') for d in diseases))
            return severities
        return []
    except Exception as e:
        st.error(f"Error loading severity levels: {str(e)}")
        return []


def get_affected_plants(api_client):
    """Get affected plants from API"""
    try:
        response = api_client.get_all_diseases()
        if response.get('success'):
            diseases = response.get('diseases', [])
            plants = set()
            for disease in diseases:
                plants.update(disease.get('affected_plants', []))
            return sorted(list(plants))
        return []
    except Exception as e:
        st.error(f"Error loading affected plants: {str(e)}")
        return []


def show_disease_content(api_client):
    """Show filtered disease content"""
    # Load diseases
    with st.spinner("Loading diseases..."):
        response = api_client.get_all_diseases()

        if not response.get('success'):
            show_error_message(f"Error loading diseases: {response.get('error', 'Unknown error')}")
            return

        diseases = response.get('diseases', [])

    if not diseases:
        show_empty_state(
            icon="🌱",
            title="No Diseases Found",
            description="The disease database is currently empty. Please check back later.",
            action_text="Refresh Database",
            action_callback=lambda: st.rerun()
        )
        return

    # Apply filters
    filters = get_current_filters()
    filtered_diseases = apply_filters_to_diseases(diseases, filters)

    # Show search results
    if filters.get('search_query'):
        st.info(f"Found {len(filtered_diseases)} diseases matching '{filters['search_query']}'")
    elif filters:
        st.info(f"Showing {len(filtered_diseases)} filtered diseases")

    # Display diseases
    if filtered_diseases:
        show_diseases_grid(filtered_diseases, api_client)

        # Show statistics
        show_disease_statistics(filtered_diseases)

        # Show severity distribution
        st.markdown("---")
        st.markdown("### 📊 Disease Severity Distribution")
        create_disease_severity_chart(filtered_diseases)
    else:
        show_empty_state(
            icon="🔍",
            title="No Diseases Match",
            description="No diseases match your current filters. Try adjusting your search criteria.",
            action_text="Clear All Filters",
            action_callback=lambda: clear_all_filters()
        )


def get_current_filters():
    """Get current filter settings"""
    return {
        'search_query': st.session_state.get('filter_search_query', ''),
        'categories': st.session_state.get('filter_categories', []),
        'severity': st.session_state.get('filter_severity', []),
        'plants': st.session_state.get('filter_plants', []),
        'contagious': st.session_state.get('filter_contagious', 'All')
    }


def clear_all_filters():
    """Clear all active filters"""
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith('filter_')]
    for key in keys_to_remove:
        del st.session_state[key]


def show_diseases_grid(diseases, api_client):
    """Display diseases in a responsive grid"""
    # Grid layout
    cols = st.columns(min(3, len(diseases)))

    for i, disease in enumerate(diseases):
        with cols[i % 3]:
            # Disease card
            with st.expander(f"🌿 {disease.get('name', 'Unknown Disease')}", expanded=False):
                show_disease_details(disease, api_client)


def show_disease_details(disease, api_client):
    """Show detailed disease information"""
    col1, col2 = st.columns([2, 1])

    with col1:
        # Basic information
        st.markdown(f"#### {disease.get('name', 'Unknown')}")
        st.markdown(f"*{disease.get('scientific_name', 'N/A')}*")

        # Category and severity badges
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Category:** {disease.get('category', 'Unknown')}")
        with col_b:
            severity = disease.get('severity', 'Unknown')
            st.markdown(f"**Severity:** {severity}")

        # Symptoms
        symptoms = disease.get('symptoms', [])
        if symptoms:
            st.markdown("**📋 Symptoms:**")
            for symptom in symptoms[:3]:  # Show first 3 symptoms
                st.markdown(f"• {symptom}")
            if len(symptoms) > 3:
                st.markdown(f"... and {len(symptoms) - 3} more")

        # Affected plants
        plants = disease.get('affected_plants', [])
        if plants:
            st.markdown("**🌱 Affected Plants:**")
            plants_text = ", ".join(plants[:5])  # Show first 5 plants
            if len(plants) > 5:
                plants_text += f" and {len(plants) - 5} more"
            st.markdown(plants_text)

    with col2:
        # Quick stats
        st.markdown("#### Quick Stats")
        st.metric("Severity Level", disease.get('severity', 'Unknown'))
        st.metric("Contagious", "Yes" if disease.get('contagious') else "No")
        st.metric("Plant Types", str(len(plants)))

        # Action buttons
        st.markdown("#### Actions")

        # View full details button
        if st.button(f"📖 Full Details", key=f"details_{disease.get('id')}"):
            # Store selected disease for detailed view
            st.session_state.selected_disease = disease
            st.session_state.current_page = "Disease Details"
            st.rerun()

        # Related diseases button
        if st.button(f"🔍 Similar Diseases", key=f"similar_{disease.get('id')}"):
            show_similar_diseases(disease, api_client)

    # Causes section
    causes = disease.get('causes', [])
    if causes:
        with st.expander("🔬 Causes & Risk Factors"):
            for cause in causes:
                st.markdown(f"• {cause}")

    # Treatment section
    treatments = disease.get('treatment', [])
    if treatments:
        with st.expander("💊 Treatment Options"):
            for treatment in treatments:
                st.markdown(f"• {treatment}")

    # Prevention section
    preventions = disease.get('prevention', [])
    if preventions:
        with st.expander("🛡️ Prevention Strategies"):
            for prevention in preventions:
                st.markdown(f"• {prevention}")


def show_similar_diseases(reference_disease, api_client):
    """Show diseases similar to reference disease"""
    disease_id = reference_disease.get('id')
    if not disease_id:
        return

    with st.spinner("Finding similar diseases..."):
        # Search for similar diseases (using search API for simplicity)
        search_response = api_client.search_diseases(reference_disease.get('name', ''), limit=5)

        if search_response.get('success') and search_response.get('diseases'):
            similar_diseases = [
                d for d in search_response.get('diseases', [])
                if d.get('id') != disease_id
            ][:4]  # Show top 4 similar diseases

            if similar_diseases:
                st.markdown("### 🔍 Similar Diseases")
                for similar_disease in similar_diseases:
                    with st.expander(f"🌿 {similar_disease.get('name', 'Unknown')}", expanded=False):
                        st.markdown(f"**{similar_disease.get('name', 'Unknown')}**")
                        st.markdown(f"*{similar_disease.get('scientific_name', 'N/A')}*")
                        st.markdown(f"**Severity:** {similar_disease.get('severity', 'Unknown')}")
                        st.markdown(f"**Category:** {similar_disease.get('category', 'Unknown')}")
            else:
                st.info("No similar diseases found")
        else:
            st.error("Unable to find similar diseases")


def show_disease_statistics(diseases):
    """Show disease database statistics"""
    st.markdown("---")
    st.markdown("### 📈 Database Statistics")

    if not diseases:
        return

    # Calculate statistics
    total_diseases = len(diseases)
    categories = list(set(d.get('category', 'Unknown') for d in diseases))
    severity_levels = list(set(d.get('severity', 'Unknown') for d in diseases))
    contagious_count = sum(1 for d in diseases if d.get('contagious', False))

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Diseases", total_diseases)

    with col2:
        st.metric("Categories", len(categories))

    with col3:
        st.metric("Severity Levels", len(severity_levels))

    with col4:
        st.metric("Contagious", contagious_count)

    # Category distribution
    category_counts = {}
    for disease in diseases:
        category = disease.get('category', 'Unknown')
        category_counts[category] = category_counts.get(category, 0) + 1

    if category_counts:
        st.markdown("#### Disease Distribution by Category")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_diseases) * 100
            st.markdown(f"**{category}:** {count} ({percentage:.1f}%)")