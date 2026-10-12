from simulation.fault_injection.communication_faults import CommunicationFaultInjector


def test_message_delivered_without_fault():
    injector = CommunicationFaultInjector(
        packet_loss_rate=0.0,
        delay_ms=0.0,
    )
    result = injector.apply({"message": "hello"})

    assert result.delivered is True
    assert result.message == {"message": "hello"}
    assert result.delay_ms == 0.0
    assert result.fault == "none"


def test_message_reports_configured_delay():
    injector = CommunicationFaultInjector(
        packet_loss_rate=0.0,
        delay_ms=50.0,
    )
    result = injector.apply({"message": "hello"})

    assert result.delivered is True
    assert result.delay_ms == 50.0
    assert result.fault == "delay"


def test_packet_loss_drops_message():
    injector = CommunicationFaultInjector(
        packet_loss_rate=1.0,
        delay_ms=0.0,
    )
    result = injector.apply({"message": "hello"})

    assert result.delivered is False
    assert result.message is None
    assert result.fault == "packet_loss"
