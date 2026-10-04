import reflex as rx
from reflex_base.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="bank_analyzer",
    plugins=[SitemapPlugin(), rx.plugins.RadixThemesPlugin()],
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="medium",
        accent_color="blue",
    ),
)
