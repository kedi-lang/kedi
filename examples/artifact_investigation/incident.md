# Checkout deployment log, 2026-09-23

This is a fictional, self-contained incident record. Times are UTC. A deployment
timestamp alone is not evidence of causality. Metrics refer to the checkout API
unless another service is named. No customer data is included.

## 08:00 - Baseline

The checkout API runs release 4.7.1 on six workers. Each worker has a database
pool of 20 connections. The database accepts 180 application connections in
total. Checkout p95 latency is 180 ms, the timeout rate is 0.02%, and the
connection-wait queue is empty. Traffic is 120 requests per second.

## 08:12 - Catalog cache maintenance

The catalog service refreshes its product cache. Its hit ratio briefly falls
from 99% to 91% and recovers by 08:15. Checkout latency remains below 200 ms.
The catalog service uses a separate database and connection pool. No checkout
requests time out during the cache refresh.

## 08:25 - Certificate rotation

The edge proxy receives a renewed TLS certificate. Health checks confirm the
new certificate chain from three regions. TLS negotiation latency remains
below 30 ms. The operation changes neither backend routing nor application
configuration. The old certificate is removed after verification.

## 08:40 - Billing export

A scheduled invoice export reads from a reporting replica. It completes in
four minutes without querying the checkout database. Replica CPU reaches 52%
and then returns to 18%. Checkout connection count and latency are unchanged.

## 09:00 - Release 4.8.0

Release 4.8.0 changes receipt formatting and adds a receipt_locale field. The
release uses the same six workers, pool size, and connection timeout as 4.7.1.
The canary serves 10% of traffic for ten minutes without a latency regression.
All workers receive 4.8.0 at 09:10. Checkout p95 remains 185 ms.

## 09:18 - Worker scaling

The worker count increases from six to eight in anticipation of a promotion.
At the existing pool size of 20, eight workers can consume 160 connections,
still below the database limit of 180. Traffic remains 120 requests per second.
The change produces no connection-wait queue and no increase in timeouts.

## 09:25 - Search index rebuild

Search begins rebuilding an index on its dedicated search cluster. The search
team reports slower indexing, not slower serving. Checkout traces contain no
search calls. Checkout database CPU remains 31%, with no lock contention.

## 09:30 - Connection pool change

A configuration rollout increases DB_POOL_SIZE from 20 to 40 on all eight
checkout workers. The database connection limit remains 180. Application
workers can now compete for up to 320 connections. This is a configuration
change separate from release 4.8.0, and rolling back application code does
not revert the environment variable.

## 09:32 - Timeout onset

The database reaches its 180-connection limit. Checkout p95 rises to 4.8 s,
the timeout rate reaches 7.2%, and traces spend 4.5 s waiting to acquire a
connection. SQL execution itself remains under 80 ms. Traffic is still 120
requests per second; there is no matching traffic surge or expensive query.

## 09:36 - First incident hypothesis

The on-call engineer suspects the recently deployed receipt formatter. The
dashboard groups application releases and configuration changes in one view,
which makes both appear near the start of the incident. The receipt formatter
does not open a second database connection; trace inspection confirms this.

## 09:42 - Application rollback

Application code rolls back from 4.8.0 to 4.7.1. DB_POOL_SIZE stays at 40.
The timeout rate remains 7.0%, p95 remains 4.7 s, and connection acquisition
still accounts for almost all of the wait. This rollback does not restore
service and does not support the receipt-formatter hypothesis.

## 09:48 - Database checks

The database team checks slow queries, locks, CPU, and storage latency. There
are no blocked transactions, CPU is 34%, and storage latency is normal.
Rejected new connection attempts rise sharply at 09:32. Established queries
continue completing in their usual time. The scarce resource is connection
admission, not query execution capacity.

## 09:55 - Configuration correction

DB_POOL_SIZE is restored to 20 on all eight workers, bringing the possible
application connection count back to 160. Workers restart in a rolling order
to discard the old pools. The database limit remains 180; no database upgrade,
traffic limit, or application code change is performed during this correction.

## 10:02 - Recovery measurement

After all workers have the corrected pool size, the connection-wait queue is
empty. Checkout p95 returns to 190 ms and the timeout rate falls to 0.03%.
Traffic remains 120 requests per second. The recovery follows the pool-size
correction, not the earlier application rollback.

## 10:20 - Release restored

Release 4.8.0 is deployed again with DB_POOL_SIZE fixed at 20. For the next
30 minutes, p95 remains between 180 and 195 ms and the timeout rate stays
below 0.04%. The receipt-formatting code is present without recreating the
incident, providing an additional check against the original hypothesis.

## 11:00 - Follow-up ownership

Platform engineering will validate the product of worker count and pool size
against the database connection budget before configuration changes. This
guard is planned, not already implemented. Observability will separate code
deployments from configuration rollouts in the incident timeline. Neither
follow-up should be described as the mitigation that restored service today.
