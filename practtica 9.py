
from wordcloud import WordCloud
import matplotlib.pyplot as plt

texto = """
Energy consumption in commercial and residential buildings is a significant concern for sustainability.
HVAC systems, lighting, occupancy patterns, and renewable energy integration play vital roles.
Temperature and humidity levels influence heating and cooling demands, especially during peak hours.
By optimizing energy usage, buildings can reduce costs and environmental impact.
Smart technologies, sensors, and predictive analytics enable real-time monitoring and efficiency.
Understanding day-of-week patterns and holidays helps in adjusting system operations.
Energy conservation strategies must also consider square footage and building usage behavior.
"""

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Energy Consumption in Buildings")
plt.show()

