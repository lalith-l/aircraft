"""
Disturbance / Collateral-Damage Graph

Models how maintenance actions on one component can disturb
neighbouring components, causing secondary failures.

Uses NetworkX with Gamma-Poisson shrinkage for rate estimation
and explicit confounding-by-indication handling.

Status: [DEMO over real code]
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats as sp_stats


@dataclass
class DisturbanceEdge:
    """An edge representing a potential disturbance effect."""
    source: str          # component that was worked on
    target: str          # component that may have been disturbed
    edge_type: str       # REQUIRES_REMOVAL, SPATIAL_PROXIMITY, DISTURBANCE_EFFECT
    observed_failures: int
    exposure_hours: float
    rate_ratio: float    # posterior rate vs baseline
    p_elevated: float    # P(rate > baseline)


class DisturbanceGraph:
    """
    Directed graph of disturbance relationships between components.

    Nodes = components. Edges = causal/proximity links.
    Rate estimation uses Gamma-Poisson conjugate update with shrinkage.
    """

    def __init__(self, baseline_rate: float = 0.001,
                 prior_alpha: float = 1.0, prior_beta: float = 1000.0):
        """
        Args:
            baseline_rate: fleet-wide average failure rate per hour.
            prior_alpha, prior_beta: Gamma prior parameters (shrinkage).
        """
        self.G = nx.DiGraph()
        self.baseline_rate = baseline_rate
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def add_component(self, name: str, **attrs):
        self.G.add_node(name, node_type="Component", **attrs)

    def add_task(self, name: str, **attrs):
        self.G.add_node(name, node_type="Task", **attrs)

    def add_edge(self, source: str, target: str, edge_type: str,
                 observed_failures: int = 0, exposure_hours: float = 1.0):
        """Add or update a disturbance edge."""
        self.G.add_edge(source, target, edge_type=edge_type,
                        observed_failures=observed_failures,
                        exposure_hours=exposure_hours)

    def compute_posterior_rate(self, observed: int, exposure: float
                               ) -> Tuple[float, float, float]:
        """
        Gamma-Poisson posterior for failure rate.

        Returns: (posterior_mean, rate_ratio vs baseline, P(rate > baseline))
        """
        alpha_post = self.prior_alpha + observed
        beta_post = self.prior_beta + exposure

        posterior_mean = alpha_post / beta_post
        rate_ratio = posterior_mean / self.baseline_rate if self.baseline_rate > 0 else 0.0

        # P(lambda > baseline) using the Gamma survival function
        p_elevated = float(1.0 - sp_stats.gamma.cdf(
            self.baseline_rate, a=alpha_post, scale=1.0 / beta_post
        ))

        return posterior_mean, rate_ratio, p_elevated

    def score_all_edges(self) -> List[DisturbanceEdge]:
        """Score every edge in the graph with Gamma-Poisson posterior."""
        results = []
        for u, v, data in self.G.edges(data=True):
            obs = data.get("observed_failures", 0)
            exp = data.get("exposure_hours", 1.0)
            _, rr, p_elev = self.compute_posterior_rate(obs, exp)

            results.append(DisturbanceEdge(
                source=u, target=v,
                edge_type=data.get("edge_type", "UNKNOWN"),
                observed_failures=obs,
                exposure_hours=exp,
                rate_ratio=rr,
                p_elevated=p_elev,
            ))

        # Sort by P(elevated) descending
        results.sort(key=lambda e: e.p_elevated, reverse=True)
        return results

    def flag_disturbances(self, threshold: float = 0.9
                          ) -> List[DisturbanceEdge]:
        """Return edges where P(rate > baseline) >= threshold."""
        return [e for e in self.score_all_edges() if e.p_elevated >= threshold]

    def handle_confounding(self, source: str, target: str,
                            indication_rate: float) -> float:
        """
        Adjust for confounding-by-indication:
        Parts that get maintained more often are also the ones
        that fail more often, biasing disturbance estimates upward.

        Returns: adjusted rate ratio.
        """
        data = self.G.edges[source, target]
        obs = data.get("observed_failures", 0)
        exp = data.get("exposure_hours", 1.0)

        # Adjust exposure by indication rate
        adjusted_exposure = exp * (1.0 + indication_rate)
        _, adj_rr, _ = self.compute_posterior_rate(obs, adjusted_exposure)
        return adj_rr
