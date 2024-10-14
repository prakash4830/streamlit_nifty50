import streamlit as st
from utils.sidebar import display_copyright

display_copyright()

col1, col2 = st.columns([1.1, 2.3])  # Adjust the column width ratio if needed

# Add the image in the first column (left side)
with col1:
    st.image("images/Nifty_50_Logo.svg", caption="Nifty 50 Index Overview", use_column_width=True)  # Customize width

# Add the title text in the second column (right side)
with col2:
    st.markdown("<h1 style='margin-top: 20px;color:#382a7f'>History and Description</h1>", unsafe_allow_html=True)

st.markdown("""
1. **Introduction to Nifty 50:**
    - The Nifty 50 is one of the leading stock market indices in India, serving as a benchmark for the Indian equity 
    market's performance. It is managed by the National Stock Exchange of India (NSE) and consists of 50 of the largest 
    and most liquid companies listed on the NSE. The index covers various sectors, reflecting the overall health of the 
    Indian economy and providing a snapshot of how large-cap companies are performing in the market.
""")
st.markdown("""
2. **History of Nifty 50:**
    - The Nifty 50 index was introduced on April 22, 1996, and was developed by India Index Services & Products Ltd. (IISL), 
a subsidiary of NSE. The aim was to create a well-diversified, transparent, and replicable index that could represent 
the top companies in India's economy. Over time, it has become a widely recognized indicator of the Indian stock market 
and is followed by both domestic and international investors.
""")

st.markdown("""
3. **Structure and Composition:**
    - Number of Constituents: The index consists of 50 large-cap companies across various sectors.
    - Sector Representation: The Nifty 50 index provides exposure to 13 sectors of the Indian economy, including 
    information technology, financial services, energy, consumer goods, and pharmaceuticals. This diversification 
    ensures that the index reflects the broad market sentiment rather than being overly reliant on any one sector.
    - Market Capitalization: It is a market-capitalization-weighted index, meaning companies with higher market 
    capitalizations have a more significant influence on the index's overall movement.
    - Rebalancing: The constituents of the Nifty 50 are reviewed and rebalanced semi-annually to ensure that it
    continues to accurately reflect the performance of the top companies in the market.
""")

st.markdown("""
4. **Performance and Importance:**
    - Performance Indicator: The Nifty 50 index is often used as a performance indicator for the Indian economy as it 
    includes companies that are leaders in their respective industries.
    - Benchmark for Investors: It serves as a benchmark for mutual funds, exchange-traded funds (ETFs), and 
    institutional investors, allowing them to gauge the performance of their portfolios relative to the broader market.
    - Growth Over Time: Since its inception, the Nifty 50 has grown in importance, reflecting the expansion of the 
    Indian economy. The index has witnessed key economic events, including the 2008 financial crisis and the economic 
    reforms of the 1990s, and has continued to grow in both value and relevance.
    - Global Recognition: The Nifty 50 is tracked by international investors looking for exposure to emerging markets, 
    making it a critical tool for global portfolio diversification.
""")

st.markdown("""
5. **Nifty 50 and the Indian Economy:**
    - Economic Barometer: As the Nifty 50 includes companies from a wide range of sectors, it is often seen as a 
    barometer of the Indian economy. When the index rises, it typically reflects investor optimism about economic 
    growth, corporate profitability, and business expansion in India.
    - Global Economic Linkages: The index's performance is also affected by global economic conditions. As demonstrated 
    by the correlation with the S&P 500, movements in the U.S. market can significantly influence the Nifty 50, 
    highlighting the interconnectedness of global financial markets.
""")

st.markdown("""
6. **Factors Affecting Nifty 50:**
    - Economic Indicators: Factors like GDP growth, inflation, and exchange rates significantly impact the index's 
    movement. A growing economy with stable inflation typically supports higher corporate earnings, which can lead to 
    rising stock prices.
    - Global Markets: Global economic conditions and international market trends, especially those of developed 
    economies like the U.S. (S&P 500), have a direct influence on the Nifty 50.
    - Government Policies: Fiscal and monetary policies, including taxation, interest rates, and regulatory reforms, 
    also affect the performance of the companies in the index.
    - Corporate Performance: The index’s performance is heavily influenced by the quarterly results, profitability, and 
    growth outlook of its constituent companies.
""")

st.markdown("""
7. **Nifty 50 Returns:** 
    - The Nifty 50 has provided investors with solid returns over the long term, making it a preferred investment 
    vehicle for long-term investors, pension funds, and institutional investors. The annualized return of the Nifty 50 
    over the past 10 years has been in the range of 10-12%, demonstrating its role as a growth-oriented index.
""")