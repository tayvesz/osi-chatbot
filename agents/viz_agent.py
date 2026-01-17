
import plotly.express as px
import pandas as pd

class VizAgent:

    def create_chart(self, dataframe, chart_type, title=""):
        if dataframe is None or dataframe.empty:
            return None
            
        try:
            if chart_type == "timeline":
                # If we have a 'count' column, line chart is great
                if 'count' in dataframe.columns.str.lower():
                    # Find year/date column
                    date_col = next((c for c in dataframe.columns if 'year' in c.lower() or 'date' in c.lower()), dataframe.columns[0])
                    count_col = next((c for c in dataframe.columns if 'count' in c.lower()), dataframe.columns[1])
                    
                    fig = px.line(
                        dataframe, x=date_col, y=count_col,
                        title=title or 'Evolution Over Time',
                        markers=True,
                        template="plotly_dark"
                    )
                else:
                    # If just a list of items with dates (Scatter plot)
                    # Find year/date column
                    date_col = next((c for c in dataframe.columns if 'year' in c.lower() or 'date' in c.lower()), None)
                    # Find text/label column (e.g., title, id)
                    label_col = next((c for c in dataframe.columns if 'title' in c.lower() or 'id' in c.lower() or 'reference' in c.lower()), dataframe.columns[0])
                    
                    if date_col:
                        fig = px.scatter(
                            dataframe, x=date_col, y=label_col,
                            title=title or 'Standards Timeline',
                            template="plotly_dark",
                            height=400 + (len(dataframe) * 20) # Auto-growth for legibility
                        )
                        fig.update_traces(marker=dict(size=10, symbol="square"))
                    else:
                         return None
                return fig
                    
            elif chart_type == "bar":
                # Smart Bar Chart
                # Try to identify category vs value
                cols = dataframe.columns
                x_col = cols[0]
                y_col = cols[1] if len(cols) > 1 else cols[0]
                
                # Check if y_col looks like a "year" (unlikely to be a value we want to sum/bar-height)
                if 'year' in y_col.lower() or 'date' in y_col.lower():
                    # Flip them if x is better? Or just don't bar chart dates as values.
                    pass 

                fig = px.bar(
                    dataframe, x=x_col, y=y_col,
                    title=title or 'Data Distribution',
                    template="plotly_dark"
                )
                return fig
                
            elif chart_type == "pie":
                names_col = dataframe.columns[0]
                values_col = dataframe.columns[1] if len(dataframe.columns) > 1 else dataframe.columns[0]
                fig = px.pie(
                    dataframe, names=names_col, values=values_col,
                    title=title or 'Composition',
                    template="plotly_dark"
                )
                return fig
                
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None
        return None

    def determine_chart_type(self, sql_query, dataframe):
        cols = dataframe.columns.str.lower()
        query_lower = sql_query.lower()
        
        # 1. Timeline / Evolution
        if any(x in cols for x in ['year', 'publication_date', 'date']):
            return "timeline"
            
        # 2. Composition (Pie) - usually for status or few categories
        if any(x in cols for x in ['status', 'type', 'stage']) and ('count' in cols or any('count' in c for c in cols)):
             if len(dataframe) < 10: # Only pie if not too many segments
                return "pie"
        
        # 3. Explicit keywords in query (e.g. from SQL)
        if 'evolution' in query_lower or 'trend' in query_lower:
             return "timeline"
             
        # 4. Comparison (Bar) - Default for counts / rankings
        # Check for count-like columns
        if any(x in c for c in cols for x in ['count', 'total', 'num', 'nb']):
            return "bar"
            
        return "bar" # Fallback
