# Kubernetes Basics

A Pod is the smallest deployable unit in Kubernetes. A Pod wraps one or more containers that share the same network namespace and storage volumes.

A Deployment manages a set of identical Pods through a ReplicaSet. It supports rolling updates, so new versions of an application are rolled out gradually without downtime, and it can roll back to a previous revision if something goes wrong.

A Service gives a stable network endpoint to a group of Pods. The ClusterIP type is reachable only inside the cluster, NodePort exposes the service on a port of every node, and LoadBalancer provisions an external load balancer from the cloud provider.

A ConfigMap stores non-sensitive configuration as key-value pairs, while a Secret stores sensitive data such as passwords and API tokens. Both can be injected into Pods as environment variables or mounted as files.

The Horizontal Pod Autoscaler adjusts the number of Pod replicas automatically based on observed CPU utilization or custom metrics.
